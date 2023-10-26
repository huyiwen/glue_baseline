from transformers import AutoModelForSequenceClassification, AutoConfig, BertTokenizer
from models.linear_mpo import MPOBertForSequenceClassification


def load_mpo_model_and_tokenizer(args):
    mpo_input_shape = {
        "word_embed": list(map(int, args.word_embed_input.split(","))),
        "FFN1": list(map(int, args.FFN1_input.split(","))),
        "FFN2": list(map(int, args.FFN2_input.split(","))),
        "attention": list(map(int, args.attention_input.split(","))),
    }
    mpo_output_shape = {
        "word_embed": list(map(int, args.word_embed_output.split(","))),
        "FFN1": list(map(int, args.FFN1_output.split(","))),
        "FFN2": list(map(int, args.FFN2_output.split(","))),
        "attention": list(map(int, args.attention_output.split(","))),
    }
    mpo_truncate_num = {
        "word_embed": 1000,
        "FFN1": 1000,
        "FFN2": 1000,
        "attention": 1000,
    }
    bert_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path.split(":")[-1])
    config = AutoConfig.from_pretrained(
        args.model_name_or_path.split(":")[-1],
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    config.mpo_layers = args.mpo_layers.split(",")
    config.mpo_input_shape = mpo_input_shape
    config.mpo_output_shape = mpo_output_shape
    config.truncate_num = mpo_truncate_num
    model = MPOBertForSequenceClassification.from_pretrained(args.model_name_or_path.split(":")[-1], config=config)
    print("#bert_p:", sum([p.numel() for p in bert_model.parameters()]))
    for param in bert_model.named_parameters():
        # print(param[0])
        setattr(model.bert, param[0], param[1].data)
    tokenizer = BertTokenizer.from_pretrained("/home/huyiwen/pretrained/bert")
    return model, tokenizer
    