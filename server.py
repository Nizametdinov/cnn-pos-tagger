from aiohttp import web
import json
import os

from model import *

def init():
    global loader, vocab, tensor_generator
    data_file = 'data/sentences.xml'
    loader = DataReader(data_file)
    loader.load()
    vocab = Vocab(loader)
    vocab.load()
    tensor_generator = TensorGenerator(loader, vocab)

    global input_, predictions, dropout
    input_, logits, dropout = model(
        max_words_in_sentence=tensor_generator.max_sentence_length,
        max_word_length=tensor_generator.max_word_length,
        char_vocab_size=vocab.char_vocab_size(),
        num_output_classes=vocab.part_vocab_size()
    )

    _targets, _target_mask, _loss_, _accuracy, predictions = loss(
        logits=logits,
        batch_size=0,
        max_words_in_sentence=tensor_generator.max_sentence_length
    )


def split_sentence(sentence_string):
    return sentence_string.split(" ")

def calculate_sentence_pos(sentence):
    with tf.Session() as session:
        restore_model(session)

        input_tensors = tensor_generator.tensor_from_sentences([sentence])
        predicted = session.run([predictions], {input_: input_tensors, dropout: 0.0})

        sentence_prediction = predicted[0][0]
        result = [[word, vocab.index_to_speech_part_human(word_prediction)] for word, word_prediction in zip(sentence, sentence_prediction)]
        return result

async def calculator(request):
    sentence_string = request.rel_url.query.get('sentence')

    if not sentence_string:
        return web.Response(status=404, text='sentence not specified')

    sentence = split_sentence(sentence_string)
    if(len(sentence) > tensor_generator.max_sentence_length):
        return web.Response(status=422, text='sentence is too long')

    result = calculate_sentence_pos(sentence)
    return web.json_response({'result': result})


##############
##   Server
##############

init()
app = web.Application()
app.router.add_route('GET', '/{tail:.*}', calculator)

# server_port = int(os.environ['SERVER_PORT']) if os.environ['SERVER_PORT'] else 3000
server_port = 8084
web.run_app(app, port=server_port)