# -*- coding: utf-8 -*-
import os

import spacy
from spacy.symbols import ORTH
import torchtext
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pcode.models.transformer as transformer
import pcode.datasets.loader.imagenet_folder as imagenet_folder
from pcode.datasets.loader.svhn_folder import define_svhn_folder
import pcode.datasets.loader.nmt_folder as nmt_folder


"""the entry for classification tasks."""


def _get_cifar(name, root, split, transform, target_transform, download):
    is_train = split == "train"

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_mnist(root, split, transform, target_transform, download):
    is_train = split == "train"

    if is_train:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    return datasets.MNIST(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_stl10(root, split, transform, target_transform, download):
    return datasets.STL10(
        root=root,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_svhn(root, split, transform, target_transform, download):
    is_train = split == "train"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return define_svhn_folder(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_imagenet(conf, name, datasets_path, split):
    is_train = split == "train"
    is_downsampled = "8" in name or "16" in name or "32" in name or "64" in name
    root = os.path.join(
        datasets_path, "lmdb" if not is_downsampled else "downsampled_lmdb"
    )

    # get transform for is_downsampled=True.
    if is_downsampled:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
                + ([normalize] if normalize is not None else [])
            )
        else:
            transform = transforms.Compose(
                [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
            )
    else:
        transform = None

    if conf.use_lmdb_data:
        if is_train:
            root = os.path.join(
                root, "{}train.lmdb".format(name + "_" if is_downsampled else "")
            )
        else:
            root = os.path.join(
                root, "{}val.lmdb".format(name + "_" if is_downsampled else "")
            )
        return imagenet_folder.define_imagenet_folder(
            conf=conf,
            name=name,
            root=root,
            flag=True,
            cuda=conf.graph.on_cuda,
            transform=transform,
            is_image=True and not is_downsampled,
        )
    else:
        assert is_downsampled
        return imagenet_folder.ImageNetDS(
            root=root, img_size=int(name[8:]), train=is_train, transform=transform
        )


"""the entry for language modeling task."""


def _get_text(batch_first):
    spacy_en = spacy.load("en")
    spacy_en.tokenizer.add_special_case("<eos>", [{ORTH: "<eos>"}])
    spacy_en.tokenizer.add_special_case("<bos>", [{ORTH: "<bos>"}])
    spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])

    def spacy_tok(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = torchtext.legacy.data.Field(
        lower=True, tokenize=spacy_tok, batch_first=batch_first
    )
    return TEXT


def _get_nlp_lm_dataset(name, datasets_path, batch_first):
    TEXT = _get_text(batch_first)

    # Load and split data.
    if "wikitext2" in name:
        train, valid, test = torchtext.datasets.WikiText2.splits(
            TEXT, root=datasets_path
        )
    elif "ptb" in name:
        train, valid, test = torchtext.datasets.PennTreebank.splits(
            TEXT, root=datasets_path
        )
    else:
        raise NotImplementedError
    return TEXT, train, valid, test


"""the entry for neural machine translation task."""


def _get_nmt_text(datasets_path, batch_first):
    tokenizer = "spacy"
    SRC_TEXT = torchtext.legacy.data.Field(
        tokenize=torchtext.data.utils.get_tokenizer(tokenizer, language="en"),
        pad_token=transformer.PAD_WORD,
        batch_first=batch_first,
    )
    TGT_TEXT = torchtext.legacy.data.Field(
        tokenize=torchtext.data.utils.get_tokenizer(tokenizer, language="de"),
        init_token=transformer.BOS_WORD,
        eos_token=transformer.EOS_WORD,
        pad_token=transformer.PAD_WORD,
        batch_first=batch_first,
    )
    return SRC_TEXT, TGT_TEXT


def _get_nlp_nmt_dataset(conf, name, datasets_path, batch_first):
    SRC_TEXT, TGT_TEXT = _get_nmt_text(datasets_path, batch_first)

    if "multi30k" in name:
        train, valid, test = nmt_folder.Multi30k.splits(
            exts=(".en", ".de"), fields=(SRC_TEXT, TGT_TEXT), root=datasets_path
        )
    elif "wmt14" in name:
        # The dataset is from Google Brain, and used by many open-sourced papers (e.g. many from fair).
        # Though this download contains test sets from 2015 and 2016,
        # the train set differs slightly from WMT 2015 and 2016 and significantly from WMT 2017.
        # train='train.tok.clean.bpe.32000', validation='newstest2013.tok.bpe.32000',
        # test='newstest2014.tok.bpe.32000'

        train, valid, test = nmt_folder.WMT14.splits(
            exts=(".en", ".de"),
            fields=(SRC_TEXT, TGT_TEXT),
            root=datasets_path,
            filter_pred=lambda x: not (
                len(vars(x)["src"]) > conf.max_sent_len
                or len(vars(x)["trg"]) > conf.max_sent_len
            ),
        )
    else:
        raise NotImplementedError

    return (SRC_TEXT, TGT_TEXT), train, valid, test


"""the entry for different supported dataset."""


def get_dataset(
    conf,
    name,
    datasets_path,
    split="train",
    transform=None,
    target_transform=None,
    download=True,
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, name)

    if name == "cifar10" or name == "cifar100":
        return _get_cifar(name, root, split, transform, target_transform, download)
    elif name == "svhn":
        return _get_svhn(root, split, transform, target_transform, download)
    elif name == "mnist":
        return _get_mnist(root, split, transform, target_transform, download)
    elif name == "stl10":
        return _get_stl10(root, split, transform, target_transform, download)
    elif "imagenet" in name:
        return _get_imagenet(conf, name, datasets_path, split)
    elif name == "wikitext2" or name == "ptb":
        return _get_nlp_lm_dataset(name, datasets_path, batch_first=False)
    elif name == "multi30k" or name == "wmt14":
        return _get_nlp_nmt_dataset(conf, name, datasets_path, batch_first=False)
    else:
        raise NotImplementedError
