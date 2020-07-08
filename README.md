# Summary

This is the code repository for the ACL 2020 paper "Extracting Headless MWEs from Dependency Parse Trees: Parsing, Tagging, and Joint Modeling Approaches".

# Dependencies

- fire
- Cython
- numpy
- pytorch
- transformers

# Data format

The data is formatted similarly as UD, except that the `MISC` column is now overided with B/I/O tags corresonding to the MWE spans.
(A future version should make this format less confusing. )

# How to train a new model

`./scripts/exec.sh $SEED $LAN $MODE $BERT $PWEIGHT`
where `$SEED` denotes a random parameter initialization,
`$LAN` is the treebank code,
`$MODE` is the training mode choosing among `parsing` `tagging` `jointparsing` `jointtagging` `jointdecoding`,
`$BERT` is a boolean value deciding whether to use pretrained bert models or not,
`$PWEIGHT` is the coefficient of parsing module weight in joint training.

Refer to the same script for other related hyperparameters used in our experiments.

# Referenced code

- https://github.com/jiesutd/NCRFpp
- https://github.com/nikitakit/self-attentive-parser
- https://github.com/allenai/allennlp

# Reference
```
@inproceedings{shi-lee-2020-extracting,
    title = "Extracting Headless {MWE}s from Dependency Parse Trees: Parsing, Tagging, and Joint Modeling Approaches",
    author = "Shi, Tianze  and
      Lee, Lillian",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.775",
    pages = "8780--8794",
}
```
