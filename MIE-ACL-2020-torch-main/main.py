import argparse
import os
from src import Dictionary, Ontology, Data, ESAL, evaluate
import json
from transformers import BertConfig,BertModel,BertTokenizer
BERT_PATH = './chinese_bert/'

parser = argparse.ArgumentParser(description='ESAL')
parser.add_argument('--add-global', type=bool, default=False, help='Add global module or not.')
parser.add_argument('--hidden-size', type=int, default=400, help='Hidden size.')
parser.add_argument('--mlp-layer-num', type=int, default=4, help='Number of layers of mlp.')
parser.add_argument('--keep-p', type=float, default=0.8, help='1 - dropout rate.')

parser.add_argument('--start-lr', type=float, default=1e-3, help='Start learning rate.')
parser.add_argument('--end-lr', type=float, default=1e-4, help='End learning rate.')
parser.add_argument('-e', '--epoch-num', type=int, default=1, help='Epoch num.')
parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size.')
parser.add_argument('-t', '--tbatch-size', type=int, default=10, help='Test batch size.')
parser.add_argument('-g', '--gpu_id', type=str, default='0', help='Gpu id.')
parser.add_argument('-l', '--location', type=str, default='model_files/ESAL', help='Location to save.')
args = parser.parse_args()

tokenizer = tokenizer = BertTokenizer.from_pretrained(BERT_PATH+"vocab.txt")
# import pdb; pdb.set_trace()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_id

ontology = Ontology(tokenizer)
ontology.add_raw('./data/ontology.json', '状态')
ontology.add_examples('./data/example_dict.json')

data = Data(100, ontology,tokenizer)
data.add_raw('train', './data/train.json', 'window')
data.add_raw('test', './data/test.json', 'window')
data.add_raw('dev', './data/dev.json', 'window')

# params of the model.
params = {
    "add_global": args.add_global,
    "num_units": args.hidden_size,
    "num_layers": args.mlp_layer_num,
    "keep_p": args.keep_p,
    'batch_size': args.batch_size,
    'gpu_ids': args.gpu_id,
    'lr': args.start_lr
}

# Initialize the model.
model = ESAL(data, ontology, params=params)

# Train the model.
model.load(args.location)

model.train(
    epoch_num=args.epoch_num,
    batch_size=args.batch_size,
    tbatch_size=args.tbatch_size,
    start_lr=args.start_lr,
    end_lr=args.end_lr,
    location=args.location)

# Test the model.
infos = evaluate(model, 'test', 100)
with open(os.path.join(args.location, 'result.json'), 'w', encoding='utf8') as f:
    json.dump(infos, f, indent=4, ensure_ascii=False)