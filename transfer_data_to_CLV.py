import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert_pretrained/chinese_L-12_H-768_A-12')
    if not os.path.exists('cached/pchatCLV/'):
        os.mkdir('cached/pchatCLV/')
    for phase in ['train', 'valid', 'test']:
        raw_data_name = 'cached/pchatkg/{}_episodes.json'.format(phase)
        with open(raw_data_name, 'r') as f:
            raw_dataset = f.readlines()
            raw_dataset = [json.loads(line) for line in raw_dataset]
        save_data_name = 'cached/pchatCLV/PChatbotW_{}.txt'.format(phase)
        with open(save_data_name, 'w') as f:
            for conversation in tqdm(raw_dataset):
                for data in conversation:
                    context = ''.join(tokenizer.convert_ids_to_tokens([int(i) for i in data['context'].split(' ')],skip_special_tokens=True))
                    response = ''.join(tokenizer.convert_ids_to_tokens([int(i) for i in data['response'].split(' ')],skip_special_tokens=True))
                    personas = [''.join(tokenizer.convert_ids_to_tokens([int(i) for i in persona.split(' ')],skip_special_tokens=True)) for persona in data['knowledge_sentences']]
                    personas_str = ''.join(personas)+'\n'
                    history_str = '<|endoftext|>'+'<|endoftext|>'.join(personas+[context])+'<|endoftext|>\n'
                    response_str = response+'<|endoftext|>\n'
                    end_str = '[SEP]\n'
                    f.write(personas_str+history_str+response_str+end_str)

