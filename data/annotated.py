'''
Data loader for annotated text datasets.
'''
import os
import re
import pdb
import enum
import glob
import array
import random
import shutil
import struct
import pickle
import tempfile
from collections import Counter
from contextlib import ExitStack

import torch
from torch import nn

import metrics
from data import preprocess
from data.text import TextDataset
from data.utils import maybe_download
from utils.file import Open, extract_all


MASKED = '<MASKED>'


class TextAnnotation(enum.Enum):
    ''' An enumeration of text annotation types '''
    NONE = ('', 'bpe.32000.bin', 'bpe.32000')

    def __init__(self, identifier, ext, vocab_ext):
        ''' Initialize the text annotation '''
        self.ext = ext
        self.vocab_ext = vocab_ext
        self.identifier = identifier

    def data_path(self, split, directory, **kwargs):
        ''' Return the data path '''
        data_ext = self.ext.format(**kwargs)
        return os.path.join(directory, f'{split}.{data_ext}')

    def vocab_path(self, directory, **kwargs):
        ''' Return the vocab path '''
        vocab_ext = self.vocab_ext.format(**kwargs)
        return os.path.join(directory, f'vocab.{vocab_ext}')


class AnnotatedTextDataset(TextDataset):
    ''' Class that encapsulates an annotated text dataset '''
    NAME = ''
    LANGUAGE_PAIR = ('en', 'en')
    WORD_COUNT = (4215814, 4186988)

    URLS = []
    RAW_SPLITS = {}
    SPLITS = {
        'train': 'train.tok',
        'valid': 'valid.tok',
        'dev': 'valid.tok',
        'test': 'test.tok'
    }

    IGNORE_REGEX_LIST = []
    SEGMENT_REGEX = re.compile(r'<\s*seg\s+id\s*=\s*"\d+"\s*>\s*(.+)\s*<\s*/\s*seg\s*>')

    def __init__(self, config, split='train', swap=False, annotation=TextAnnotation.NONE):
        ''' Initialize the annotated text dataset '''
        super(AnnotatedTextDataset, self).__init__(config, split=split)

        self.swap = swap
        self.segmenters = []
        self.annotation = annotation

    @classmethod
    def name(cls, swap=False, annotation=TextAnnotation.NONE):
        ''' Return a name for the dataset given the passed in configuration '''
        config = [cls.NAME] + list(reversed(cls.LANGUAGE_PAIR) if swap else cls.LANGUAGE_PAIR)
        if annotation.identifier:
            config += [annotation.identifier]

        return '_'.join(config)

    @property
    def source_language(self):
        ''' Return the source language '''
        return type(self).LANGUAGE_PAIR[1 if self.swap else 0]

    @property
    def target_language(self):
        ''' Return the target language '''
        return type(self).LANGUAGE_PAIR[0 if self.swap else 1]

    @property
    def word_count_ratio(self):
        ''' Return the word count ratio between source and target languages '''
        return type(self).WORD_COUNT[1] / type(self).WORD_COUNT[0] if self.swap \
            else type(self).WORD_COUNT[0] / type(self).WORD_COUNT[1]

    @property
    def mask_idx(self):
        ''' Return the start of summary value '''
        return self.token2id[MASKED]

    @property
    def base_data_path(self):
        ''' Get the path of the processed data file '''
        return TextAnnotation.NONE.data_path(
            type(self).SPLITS[self.split],
            self.preprocess_directory
        )

    @property
    def source_annotation_data_path(self):
        ''' Get the path of the processed data file '''
        return self.annotation.data_path(
            type(self).SPLITS[self.split],
            self.preprocess_directory,
            lang=self.source_language
        )

    @property
    def target_annotation_data_path(self):
        ''' Get the path of the processed data file '''
        return self.annotation.data_path(
            type(self).SPLITS[self.split],
            self.preprocess_directory,
            lang=self.target_language
        )

    @property
    def data_paths(self):
        ''' Get the list of data files '''
        return set([
            self.base_data_path,
            self.source_annotation_data_path,
            self.target_annotation_data_path
        ])

    @property
    def base_vocab_path(self):
        ''' Get the path of the vocab file '''
        return TextAnnotation.NONE.vocab_path(
            self.preprocess_directory
        )

    @property
    def annotation_vocab_path(self):
        ''' Get the path of the annotation specific vocab file '''
        return self.annotation.vocab_path(
            self.preprocess_directory
        )

    @property
    def vocab_paths(self):
        ''' Get the list of vocab files '''
        return set([self.base_vocab_path, self.annotation_vocab_path])

    @property
    def preprocess_directory(self):
        ''' Get the preprocess directory '''
        return self.config.preprocess_directory

    @property
    def preprocess_buffer_size(self):
        ''' Get the preprocess buffer size '''
        return self.config.preprocess_buffer_size

    def collate_field(self, batch, field_name, values):
        ''' Collate a specific field '''
        if 'annotation' in field_name:
            batch[field_name + 's'] = nn.utils.rnn.pad_sequence(
                values, batch_first=True, padding_value=self.padding_idx - self.reserved_range)
            batch[field_name + '_lens'] = torch.LongTensor([len(sequence) for sequence in values])
        else:
            super(AnnotatedTextDataset, self).collate_field(batch, field_name, values)

    def preprocess_raw_line(self, line, xml=False):
        ''' Preprocess the raw text '''
        line = line.strip()
        if self.config.max_line_length and len(line) > self.config.max_line_length:
            return

        if any(ignore.match(line) for ignore in type(self).IGNORE_REGEX_LIST):
            return

        if xml:
            match = type(self).SEGMENT_REGEX.match(line)
            if not match:
                return
            return match[1]

        return line

    def download_and_extract(self):
        ''' Download and extract the dataset '''
        for filename, url in type(self).URLS:
            filepath = os.path.join(self.config.data_directory, filename)
            maybe_download(filepath, url)
            extract_all(filepath, self.preprocess_directory)

    def preprocess_raw(self):
        ''' Tokenize/bpe encode the raw text '''
        def is_xml(filename):
            ''' Determine if a file is XML formatted '''
            return filename.endswith('.sgm') or filename.endswith('.xml')

        def filter_lines(in_file, basename):
            ''' Scan the file for any filtered lines '''
            filtered = set()
            xml = is_xml(basename)
            for i, line in enumerate(in_file):
                if not self.preprocess_raw_line(line, xml=xml):
                    filtered.add(i)

            return filtered

        def merge(basename, in_file, out_file, filtered=None):
            ''' Tokenize the passed in file and write it to the designated file '''
            filtered = filtered or set()
            xml = is_xml(basename)
            for i, line in enumerate(in_file):
                if i in filtered:
                    continue

                processed_line = self.preprocess_raw_line(line, xml=xml)
                out_file.write(processed_line + '\n')

        # First, clean-up any incomplete preprocessing files
        for path in glob.glob(os.path.join(self.preprocess_directory, '*.incomplete')):
            os.remove(os.path.join(self.preprocess_directory, path))

        bpe_code_path = os.path.join(self.preprocess_directory, 'bpe.32000')
        if not os.path.exists(bpe_code_path):
            for split, file_pairs in type(self).RAW_SPLITS.items():
                for pair in file_pairs:
                    # First determine which lines must be skipped in both files, since the files are
                    # a parallel corpora.
                    filtered = set()
                    for filename, lang in zip(pair, type(self).LANGUAGE_PAIR):
                        in_path = os.path.join(self.preprocess_directory, filename)
                        with ExitStack() as stack:
                            in_file = stack.enter_context(Open(in_path, 'rt'))
                            filtered.update(filter_lines(in_file, os.path.basename(filename)))

                    for filename, lang in zip(pair, type(self).LANGUAGE_PAIR):
                        basename = os.path.basename(filename)
                        in_path = os.path.join(self.preprocess_directory, filename)
                        split_path = os.path.join(self.preprocess_directory, f'{split}.{lang}')

                        if os.path.exists(split_path):
                            continue

                        with ExitStack() as stack:
                            out_path = f'{split_path}.incomplete'
                            in_file = stack.enter_context(Open(in_path, 'rt'))
                            out_file = stack.enter_context(Open(out_path, 'at'))

                            merge(basename, in_file, out_file, filtered)

            word_counts = Counter()
            for split in type(self).RAW_SPLITS:
                for lang in type(self).LANGUAGE_PAIR:
                    try:
                        split_path = os.path.join(self.preprocess_directory, f'{split}.{lang}')
                        os.rename(f'{split_path}.incomplete', split_path)
                    except FileNotFoundError:
                        # This can happen if the preprocessing is interrupted
                        pass

                    tokenized_path = os.path.join(self.preprocess_directory, f'{split}.tok.{lang}')
                    word_counts.update(preprocess.tokenize(
                        split_path, tokenized_path, self.preprocess_buffer_size
                    ))

            print('Learning BPE')
            preprocess.learn_bpe(bpe_code_path, word_counts.items())

        vocab_path = os.path.join(self.preprocess_directory, 'vocab.bpe.32000')
        if not os.path.exists(vocab_path):
            vocab = set()
            for split in type(self).RAW_SPLITS:
                for lang in type(self).LANGUAGE_PAIR:
                    in_path = os.path.join(
                        self.preprocess_directory,
                        f'{split}.tok.{lang}'
                    )
                    bpe_path = os.path.join(
                        self.preprocess_directory,
                        f'{split}.tok.bpe.32000.{lang}'
                    )

                    vocab.update(preprocess.apply_bpe(
                        bpe_code_path, in_path, bpe_path, self.preprocess_buffer_size
                    ))

            vocab_path = os.path.join(self.preprocess_directory, 'vocab.bpe.32000')
            incomplete_vocab_path = f'{vocab_path}.incomplete'
            with Open(incomplete_vocab_path, 'wt') as vocab_file:
                vocab_file.writelines('\n'.join([word for word in sorted(vocab)]))
            os.rename(incomplete_vocab_path, vocab_path)

    def preprocess(self):
        ''' Do any data preprocessing if needed '''
        if (
                all(os.path.exists(p) for p in self.data_paths) and
                all(os.path.exists(p) for p in self.vocab_paths)
        ):
            return

        if not os.path.exists(self.preprocess_directory):
            os.makedirs(self.preprocess_directory)

        self.download_and_extract()
        self.preprocess_raw()

        # Make sure we have loaded the vocab
        self.load_vocab(preprocessing=True)

        split_filename = type(self).SPLITS[self.split]
        self.preprocess_bpe(split_filename)

    def preprocess_bpe(self, filename):
        ''' Preprocess the BPE data '''
        tokenized_bpe_path = os.path.join(self.preprocess_directory, f'{filename}.bpe.32000')

        target_path = f'{tokenized_bpe_path}.{self.target_language}'
        source_path = f'{tokenized_bpe_path}.{self.source_language}'
        processed_path = f'{tokenized_bpe_path}.bin'

        if os.path.exists(processed_path):
            return

        with ExitStack() as stack:
            source_file = stack.enter_context(Open(source_path, 'rt'))
            target_file = stack.enter_context(Open(target_path, 'rt'))

            def encode_sentence(line):
                ''' Helper function that encodes a sentence '''
                sentence = array.array('H')
                sentence.extend((
                    self.token2id[token]
                    for token in line.split()
                ))

                byte_rep = sentence.tostring()
                byte_len = len(byte_rep)
                return struct.pack('Q{}s'.format(byte_len), byte_len, byte_rep)

            out_file = stack.enter_context(tempfile.NamedTemporaryFile())
            for source_line, target_line in zip(source_file, target_file):
                source_sentence = encode_sentence(source_line)
                target_sentence = encode_sentence(target_line)

                out_file.write(source_sentence)
                out_file.write(target_sentence)

            out_file.flush()
            shutil.copy(out_file.name, f'{processed_path}.incomplete')
            os.rename(f'{processed_path}.incomplete', processed_path)

    def load_vocab(self, preprocessing=False):
        ''' Return the data loader for the dataset '''
        if not os.path.exists(self.base_vocab_path):
            print('Cannot find the vocab file!')
            exit(1)

        with Open(self.base_vocab_path, 'rt') as vocab_file:
            self.token2id = {}
            self.id2token = []
            for token in vocab_file.read().split('\n'):
                self.token2id[token] = len(self.id2token)
                self.id2token.append(token)

        super(AnnotatedTextDataset, self).load_vocab(preprocessing)

    def load_text(self):
        ''' Load the translations '''
        if not all(os.path.exists(p) for p in self.data_paths):
            print('Cannot find the processed translations!')
            exit(1)

        with ExitStack() as stack:
            base_data_file = stack.enter_context(Open(self.base_data_path, 'rb'))

            while True:
                if self.swap:
                    source_key = 'target'
                    target_key = 'input'
                else:
                    source_key = 'input'
                    target_key = 'target'

                example = {}
                example['input'] = array.array('H')
                example['target'] = array.array('H')

                # prepend the start of sentence token to the target
                example['target'].append(self.sos_idx)

                source_sentence_len = base_data_file.read(8)
                if not source_sentence_len:
                    break

                source_sentence_len, = struct.unpack('Q', source_sentence_len)
                example[source_key].fromstring(base_data_file.read(source_sentence_len))

                target_sentence_len = base_data_file.read(8)
                if not target_sentence_len:
                    print('Unexpected end of file while trying to read a de sentence!')
                    exit(1)

                target_sentence_len, = struct.unpack('Q', target_sentence_len)
                example[target_key].frombytes(base_data_file.read(target_sentence_len))

                # append the end of sentence token to the target
                example['target'].append(self.eos_idx)

                if example == {}:
                    return
                self.add_datum(example)
