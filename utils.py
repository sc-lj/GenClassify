from dataclasses import dataclass
from typing import List, Counter, Tuple, Dict
import json
from collections import defaultdict
from typing import List
import numpy as np
import re
from nltk.tree import ParentedTree
import torch
from collections import OrderedDict
import math
import random
from typing import Optional, Union
from collections import OrderedDict
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.file_utils import PaddingStrategy
from transformers import BertTokenizer

# 标签的开始标志
TYPE_START = '<extra_id_0>'
# 标签的结束标志
TYPE_END = '<extra_id_1>'
# 输入文本的开始标志
TEXT_START = '<extra_id_2>'
# 文本span的开始标志
SPAN_START = '<extra_id_5>'
# 非文本span的标志
null_span = '<extra_id_6>'
null_label = '<extra_id_7>'

left_bracket = '【'
right_bracket = '】'
brackets = left_bracket + right_bracket

split_bracket = re.compile(r"<extra_id_\d>")


class BaseStructureMarker():
    def __init__(self) -> None:
        super().__init__()
        self.sent_start = '<extra_id_0>'
        self.sent_end = '<extra_id_1>'
        self.record_start = '<extra_id_0>'
        self.record_end = '<extra_id_1>'
        self.span_start = '<extra_id_0>'
        self.span_end = '<extra_id_1>'
        self.text_start = '<extra_id_2>'
        self.source_span_start = '<extra_id_3>'
        self.source_span_end = '<extra_id_4>'
        self.target_span_start = '<extra_id_5>'
        self.null_span = '<extra_id_6>'
        self.null_label = '<extra_id_7>'



class RecordSchema:
    def __init__(self, type_list, role_list):
        self.type_list = type_list
        self.role_list = role_list

    def __repr__(self) -> str:
        return f"Type: {self.type_list}\n Role: {self.role_list}\n"

    @staticmethod
    def get_empty_schema():
        return RecordSchema(type_list=list(), role_list=list())

    @staticmethod
    def read_from_file(filename):
        lines = open(filename).readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        return RecordSchema(type_list, role_list)

    def write_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write(json.dumps(self.type_list) + '\n')
            output.write(json.dumps(self.role_list) + '\n')


def merge_schema(schema_list: List[RecordSchema]):
    type_set = set()
    role_set = set()

    for schema in schema_list:
        for type_name in schema.type_list:
            type_set.add(type_name)

        for role_name in schema.role_list:
            role_set.add(role_name)

    return RecordSchema(type_list=list(type_set),
                        role_list=list(role_set)
                        )



class PredictParser:
    def __init__(self, label_constraint=None):
        self.spot_set = label_constraint.type_list if label_constraint else list()
        self.role_set = label_constraint.role_list if label_constraint else list()

    def decode(self, gold_list, pred_list, text_list=None, raw_list=None) -> Tuple[List, Counter]:
        pass


class SpotAsocPredictParser(PredictParser):
    def decode(self, gold_list, pred_list, text_list=None, raw_list=None) -> Tuple[List[Dict], Counter]:
        """
        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_spot -> [(type1, text1), (type2, text2), ...]
                gold_spot -> [(type1, text1), (type2, text2), ...]
                pred_asoc -> [(spot type1, asoc type1, text1), (spot type2, asoc type2, text2), ...]
                gold_asoc -> [(spot type1, asoc type1, text1), (spot type2, asoc type2, text2), ...]
                pred_record -> [{'type': type1, 'text': text1, 'roles': [(spot type1, asoc type1, text1), ...]},
                                {'type': type2, 'text': text2, 'roles': [(spot type2, asoc type2, text2), ...]},
                                ]
                gold_record -> [{'type': type1, 'text': text1, 'roles': [(spot type1, asoc type1, text1), ...]},
                                {'type': type2, 'text': text2, 'roles': [(spot type2, asoc type2, text2), ...]},
                                ]
            Counter:
        """
        counter = Counter()
        well_formed_list = []

        if gold_list is None or len(gold_list) == 0:
            gold_list = ["%s%s" % (TYPE_START, TYPE_END)] * len(pred_list)

        if text_list is None:
            text_list = [None] * len(pred_list)

        if raw_list is None:
            raw_list = [None] * len(pred_list)

        for gold, pred, text, raw_data in zip(gold_list, pred_list, text_list,
                                              raw_list):
            gold = convert_bracket(gold)
            pred = convert_bracket(pred)

            pred = clean_text(pred)

            try:
                gold_tree = ParentedTree.fromstring(gold, brackets=brackets)
            except ValueError:
                gold_tree = ParentedTree.fromstring(add_bracket(gold), brackets=brackets)
                counter.update(['gold_tree add_bracket'])

            instance = {
                'gold': gold,
                'pred': pred,
                'gold_tree': gold_tree,
                'text': text,
                'raw_data': raw_data
            }

            counter.update(['gold_tree' for _ in gold_tree])

            instance['gold_spot'], instance['gold_asoc'], instance['gold_record'] = self.get_record_list(
                sel_tree=instance["gold_tree"],
                text=instance['text']
            )

            try:
                if not check_well_form(pred):
                    pred = add_bracket(pred)
                    counter.update(['fixed'])

                pred_tree = ParentedTree.fromstring(pred, brackets=brackets)
                counter.update(['pred_tree' for _ in pred_tree])

                instance['pred_tree'] = pred_tree
                counter.update(['well-formed'])

            except ValueError:
                counter.update(['ill-formed'])
                instance['pred_tree'] = ParentedTree.fromstring(
                    left_bracket + right_bracket,
                    brackets=brackets
                )

            instance['pred_spot'], instance['pred_asoc'], instance['pred_record'] = self.get_record_list(
                sel_tree=instance["pred_tree"],
                text=instance['text']
            )

            well_formed_list += [instance]

        return well_formed_list, counter

    def get_record_list(self, sel_tree, text=None):
        """ Convert single sel expression to extraction records
        Args:
            sel_tree (Tree): sel tree
            text (str, optional): _description_. Defaults to None.
        Returns:
            spot_list: list of (spot_type: str, spot_span: str)
            asoc_list: list of (spot_type: str, asoc_label: str, asoc_text: str)
            record_list: list of {'asocs': list(), 'type': spot_type, 'spot': spot_text}
        """

        spot_list = list()
        asoc_list = list()
        record_list = list()

        for spot_tree in sel_tree:

            # Drop incomplete tree
            if isinstance(spot_tree, str) or len(spot_tree) == 0:
                continue

            spot_type = spot_tree.label()
            spot_text = get_tree_str(spot_tree)
            spot_type, spot_text = resplit_label_span(
                spot_type, spot_text)
            spot_type, spot_text = rewrite_label_span(
                label=spot_type,
                span=spot_text,
                label_set=self.spot_set,
                text=text
            )

            # Drop empty generated span
            if spot_text is None or spot_text == null_span:
                continue
            # Drop empty generated type
            if spot_type is None:
                continue
            # Drop invalid spot type
            if self.spot_set is not None and spot_type not in self.spot_set:
                continue

            record = {'asocs': list(),
                      'type': spot_type,
                      'spot': spot_text}

            for asoc_tree in spot_tree:
                if isinstance(asoc_tree, str) or len(asoc_tree) < 1:
                    continue

                asoc_label = asoc_tree.label()
                asoc_text = get_tree_str(asoc_tree)
                asoc_label, asoc_text = resplit_label_span(
                    asoc_label, asoc_text)
                asoc_label, asoc_text = rewrite_label_span(
                    label=asoc_label,
                    span=asoc_text,
                    label_set=self.role_set,
                    text=text
                )

                # Drop empty generated span
                if asoc_text is None or asoc_text == null_span:
                    continue
                # Drop empty generated type
                if asoc_label is None:
                    continue
                # Drop invalid asoc type
                if self.role_set is not None and asoc_label not in self.role_set:
                    continue

                asoc_list += [(spot_type, asoc_label, asoc_text)]
                record['asocs'] += [(asoc_label, asoc_text)]

            spot_list += [(spot_type, spot_text)]
            record_list += [record]

        return spot_list, asoc_list, record_list


def add_space(text):
    """
    add space between special token
    """
    new_text_list = list()
    for item in zip(split_bracket.findall(text), split_bracket.split(text)[1:]):
        new_text_list += item
    return ' '.join(new_text_list)


def convert_bracket(text):
    text = add_space(text)
    for start in [TYPE_START]:
        text = text.replace(start, left_bracket)
    for end in [TYPE_END]:
        text = text.replace(end, right_bracket)
    return text


def find_bracket_num(tree_str):
    """
    Count Bracket Number (num_left - num_right), 0 indicates num_left = num_right
    """
    count = 0
    for char in tree_str:
        if char == left_bracket:
            count += 1
        elif char == right_bracket:
            count -= 1
        else:
            pass
    return count


def check_well_form(tree_str):
    return find_bracket_num(tree_str) == 0


def clean_text(tree_str):
    count = 0
    sum_count = 0

    tree_str_list = tree_str.split()

    for index, char in enumerate(tree_str_list):
        if char == left_bracket:
            count += 1
            sum_count += 1
        elif char == right_bracket:
            count -= 1
            sum_count += 1
        else:
            pass
        if count == 0 and sum_count > 0:
            return ' '.join(tree_str_list[:index + 1])
    return ' '.join(tree_str_list)


def resplit_label_span(label, span, split_symbol=SPAN_START):
    label_span = label + ' ' + span

    if split_symbol in label_span:
        splited_label_span = label_span.split(split_symbol)
        if len(splited_label_span) == 2:
            return splited_label_span[0].strip(), splited_label_span[1].strip()

    return label, span


def add_bracket(tree_str):
    """add right bracket to fix ill-formed expression
    """
    tree_str_list = tree_str.split()
    bracket_num = find_bracket_num(tree_str_list)
    tree_str_list += [right_bracket] * bracket_num
    return ' '.join(tree_str_list)


def get_tree_str(tree):
    """get str from sel tree
    """
    str_list = list()
    for element in tree:
        if isinstance(element, str):
            str_list += [element]
    return ' '.join(str_list)


def rewrite_label_span(label, span, label_set=None, text=None):

    # Invalid Type
    if label_set and label not in label_set:
        return None, None

    # Fix unk using Text
    if text is not None and '<unk>' in span:
        span = fix_unk_from_text(span, text, '<unk>')

    # Invalid Text Span
    if text is not None and span not in text:
        return None, None

    return label, span


def fix_unk_from_text(span, text, unk='<unk>'):
    """
    Find span from the text to fix unk in the generated span
    从 text 中找到 span，修复span

    Example:
    span = "<unk> colo e Bengo"
    text = "At 159 meters above sea level , Angola International Airport is located at Ícolo e Bengo , part of Luanda Province , in Angola ."

    span = "<unk> colo e Bengo"
    text = "Ícolo e Bengo , part of Luanda Province , in Angola ."

    span = "Arr<unk> s negre"
    text = "The main ingredients of Arròs negre , which is from Spain , are white rice , cuttlefish or squid , cephalopod ink , cubanelle and cubanelle peppers . Arròs negre is from the Catalonia region ."

    span = "colo <unk>"
    text = "At 159 meters above sea level , Angola International Airport is located at e Bengo , part of Luanda Province , in Angola . coloÍ"

    span = "Tarō As<unk>"
    text = "The leader of Japan is Tarō Asō ."

    span = "Tar<unk> As<unk>"
    text = "The leader of Japan is Tarō Asō ."

    span = "<unk>Tar As<unk>"
    text = "The leader of Japan is ōTar Asō ."
    """
    if unk not in span:
        return span

    def clean_wildcard(x):
        sp = ".*?()[]+"
        return re.sub("("+"|".join([f"\\{s}" for s in sp])+")", "\\\\\g<1>", x)

    match = r'\s*\S+\s*'.join([clean_wildcard(item.strip()) for item in span.split(unk)])

    result = re.search(match, text)

    if not result:
        return span
    return result.group().strip()


@dataclass
class SpotAsocNoiser:
    spot_noise_ratio: float = 0.1
    asoc_noise_ratio: float = 0.1
    null_span: str = null_span

    def random_insert_spot(self, spot_asoc, spot_label_list=None):
        """随机插入 Spot，类别从 spot_label_list 中自动选择

        Args:
            spot_asoc ([type]): [description]
            spot_label_list ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if spot_label_list is None or len(spot_label_list) == 0:
            return spot_asoc
        random_num = sum(np.random.binomial(1, self.spot_noise_ratio, len(spot_asoc)))
        for _ in range(random_num):
            random_position = np.random.randint(low=0, high=len(spot_asoc))
            random_label = np.random.choice(spot_label_list)
            spot_asoc.insert(
                random_position,
                {"span": self.null_span, "label": random_label, 'asoc': list()}
            )
        return spot_asoc

    def random_insert_asoc(self, spot_asoc, asoc_label_list=None):
        """随机插入 Asoc，类别从 asoc_label_list 中自动选择

        Args:
            spot_asoc ([type]): [description]
            asoc_label_list ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if asoc_label_list is None or len(asoc_label_list) == 0:
            return spot_asoc
        # asoc_sum = sum([len(x['asoc']) for x in spot_asoc])
        spot_sum = len(spot_asoc)
        random_num = sum(np.random.binomial(1, self.asoc_noise_ratio, spot_sum))
        for _ in range(random_num):
            random_label = np.random.choice(asoc_label_list)
            spot_position = np.random.randint(low=0, high=len(spot_asoc))
            asoc_position = np.random.randint(low=0, high=len(spot_asoc[spot_position]['asoc']) + 1)
            spot_asoc[spot_position]['asoc'].insert(
                asoc_position,
                (random_label, self.null_span)
            )
        return spot_asoc

    def add_noise(self, spot_asoc, spot_label_list, asoc_label_list):
        spot_asoc = self.random_insert_asoc(
            spot_asoc=spot_asoc,
            asoc_label_list=asoc_label_list,
        )
        spot_asoc = self.random_insert_spot(
            spot_asoc=spot_asoc,
            spot_label_list=spot_label_list,
        )
        return spot_asoc



@dataclass
class DataCollatorForMetaSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        max_target_length (:obj:`int`, `optional`):
            Maximum length of target sequence length.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    max_target_length: Optional[int] = None
    max_prefix_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    spot_asoc_nosier: SpotAsocNoiser = None

    def __call__(self, features):
        """ Make Meta Schema Batch

        Args:
            features (Dict): [description]
                - input_ids
                - attention_mask
                - labels

        Returns:
        """
        for feature in features:
            if self.max_length:
                feature['input_ids'] = feature['input_ids'][:self.max_length]
            if self.max_target_length and 'labels' in feature:
                feature['labels'] = feature['labels'][:self.max_target_length]

            feature['attention_mask'] = [1] * len(feature['input_ids'])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(_label) for _label in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"])

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


def convert_spot_asoc(spot_asoc_instance, structure_maker):
    """将一个 Spot-Asoc 实例转换成目标字符串

    Args:
        spot_asoc_instance ([type]): [description]
        structure_maker ([type]): [description]

    Returns:
        [type]: [description]
    """
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['label'],
            structure_maker.target_span_start,
            spot['span'],
        ]
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_maker.span_start,
                asoc_label,
                structure_maker.target_span_start,
                asoc_span,
                structure_maker.span_end,
            ]
            spot_str_rep += [' '.join(asoc_str_rep)]
        spot_instance_str_rep_list += [' '.join([
            structure_maker.record_start,
            ' '.join(spot_str_rep),
            structure_maker.record_end,
        ])]
    target_text = ' '.join([
        structure_maker.sent_start,
        ' '.join(spot_instance_str_rep_list),
        structure_maker.sent_end,
    ])
    return target_text


def convert_spot_asoc_name(spot_asoc_instance, structure_maker):
    """将一个 Spot-Asoc-Name 实例转换成目标字符串

    Args:
        spot_asoc_instance ([type]): [description]
        structure_maker ([type]): [description]

    Returns:
        [type]: [description]
    """
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['span'],
            structure_maker.target_span_start,
            spot['label'],
        ]
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_maker.span_start,
                asoc_span,
                structure_maker.target_span_start,
                asoc_label,
                structure_maker.span_end,
            ]
            spot_str_rep += [' '.join(asoc_str_rep)]
        spot_instance_str_rep_list += [' '.join([
            structure_maker.record_start,
            ' '.join(spot_str_rep),
            structure_maker.record_end,
        ])]
    target_text = ' '.join([
        structure_maker.sent_start,
        ' '.join(spot_instance_str_rep_list),
        structure_maker.sent_end,
    ])
    return target_text


def get_label_name_tree(label_name_list, tokenizer, end_symbol='<end>'):
    """构建类型列表树
    Args:
        label_name_list ([type]): [实体的类型列表或者角色的类型列表]
        tokenizer ([type]): [description]
        end_symbol ([type]): [结束符号标志]
    Returns:
        [type]: [description]
    """
    sub_token_tree = dict()

    label_tree = dict()
    for typename in label_name_list:
        after_tokenized = tokenizer.encode(typename, add_special_tokens=False)
        # label_tree[typename] = tokenizer.convert_ids_to_tokens(after_tokenized)
        label_tree[typename] = after_tokenized

    for _, sub_label_seq in label_tree.items():
        parent = sub_token_tree
        for value in sub_label_seq:
            if value not in parent:
                parent[value] = dict()
            parent = parent[value]

        parent[end_symbol] = None

    return sub_token_tree


class T5BertTokenizer(BertTokenizer):

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self,
                 vocab_file,
                 do_lower_case=False,
                 do_basic_tokenize=True,
                 never_split=None,
                 unk_token="<unk>",
                 sep_token=None,
                 pad_token="<pad>",
                 cls_token=None,
                 mask_token=None,
                 space_token="<space>",
                 tokenize_chinese_chars=True,
                 strip_accents=None,
                 **kwargs):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs, )

        self._space_token = space_token

    def get_vocab(self):
        vocab = {
            self.convert_ids_to_tokens(i): i
            for i in range(self.vocab_size)
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def tokenize(self, text):
        import re
        # Remove space between <extra_id_*> <spot> <asoc>
        split_bracket = re.compile(
            r"\s*<extra_id_\d>\s*|\s*<spot>\s*|\s*<asoc>\s*")

        if len(split_bracket.split(text)) > 1:
            new_text_list = [split_bracket.split(text)[0]]
            for item in zip(
                    split_bracket.findall(text), split_bracket.split(text)[1:]):
                new_text_list += [item[0].strip(), item[1]]
            text = "".join(new_text_list)
        text = text.replace(' ', self._space_token)
        return super().tokenize(text)

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: ``X </s>``
        - pair of sequences: ``A </s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def _decode(self,
                token_ids: List[int],
                skip_special_tokens: bool = False,
                **kwargs) -> str:
        tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens)

        # Fix '##' subtoken
        tokens = [x.lstrip('#') if x.startswith("##") else x for x in tokens]

        x_str = "".join(tokens)
        x_str = x_str.replace(' ', '')
        x_str = x_str.replace(self._space_token, ' ')
        return x_str



def update_arguments(args, config):
    """将config中的参数更新到args中
    Args:
        args ([type]): [description]
        config ([type]): [description]
    """
    for key, value in config.items():
        # 对于args中设置的值为最终值,即使config里面有冲突的值,仍以args中的参数值为准
        if key in args:
            print(f"该参数{key}的原值为{value},新值为{args.__dict__[key]}")
            continue
        args.__setattr__(key, value)
    return args