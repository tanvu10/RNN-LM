import sys
import time
import os

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample
from utils import weights_init

import torch
import torch.nn as nn
from torch.autograd import Variable


class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  ### YOUR CODE HERE
  batch_size = 64
  embed_size = 50
  hidden_size = 100
  num_steps = 10 # RNN is unfolded into 'num_steps' time steps for training
  max_epochs = 1 # the number of max epoch
  early_stopping = 2
  dropout = 0.1
  lr = 0.01
  vocab_size= 0
  ### END YOUR CODE

class RNNLM_Model(nn.Module):
    
  def __init__(self, config):
    """Initialize the model."""
    super(RNNLM_Model, self).__init__()
    self.config = config

    ### YOUR CODE HERE
    ### Define the Embedding layer. Hint: check nn.Embedding

    ### Define the H, I, b1 in HW4. Hint: check nn.Parameter

    ### Define the projection layer, U, b2 in HW4

    ## Define the input dropout and output dropout.
    self.input_drop = 
    self.output_drop = 
    ### END YOUR CODE

    ## Initialize the weights. 
    weights_init(self)
    

  def forward(self, input_x, initial_state):
    """Build the model."""
    ### You do not need to modify this forward function
    ### you need to modify the following three functions 
    #First, we need to perform embedding lookup using the embedding matrix.
    input_x = self.add_embedding(input_x)
    #Next, compute the hidden states for different steps 
    rnn_outputs, last_state = self.add_model(input_x, initial_state)
    #Compute the prediction of different steps 
    outputs = self.add_projection(rnn_outputs)
    return outputs, final_state
    

  def add_embedding(self, input_x):
    """Add embedding layer.

    Hint: Please refer to torch.nn.Embedding

    Hint: You might find torch.split, torch.squeeze useful in constructing tensor inputs.

    Hint: embedding:  corresponding to L in HW4.

    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """
    ### YOUR CODE HERE

    ### END YOUR CODE
    return input_x

  def add_model(self, input_x, initial_state):
    """Creates the RNN language model.

    Implement the equations for the RNN language model.
    Note that you CANNOT use built in rnn_cell from torch library.

    Hint: Make sure to apply dropout to both the inputs and the outputs.
          How to do it for inputs has been provided.

    Hint: To implement RNN, you need to perform an explicit for-loop over inputs.
          For the first step, take the initial_state as the input hidden state. 
          For following steps, take the previous output state as the input hidden state.

    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
               The final state in this batch, defined as the final_state
    """
    input_x = [self.input_drop(x) for x in input_x]

    ### YOUR CODE HERE

    ### END YOUR CODE
    return rnn_outputs, final_state


  def add_projection(self, rnn_outputs):
    """Adds a projection/output layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab))
    """
    ### YOUR CODE HERE

    ### END YOUR CODE
    return outputs

  def init_hidden(self):
    """
        Hint: Use a zeros tensor of shape (batch_size, hidden_size) as
          initial state for the RNN. You might find torch.zeros useful.    
        Hint: If you are using GPU, the init_hidden should be attached to cuda.
    """
    ### YOUR CODE HERE

    ### END YOUR CODE
    return init_state
    

def generate_text(model, config, vocab,  starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
  """Generate text from the model. Note that batch_size and num_steps are both 1.

  Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
  Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
  Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)
  Hint: Dropout rate should be 1 for this work.

  Args:
    model: Object of type RNNLM_Model
    config: A Config() object
    vocab: the vocab
    starting_text: Initial text passed to model.
  Returns:
    output: List of words
  """
  model.eval()
  state = model.init_hidden()
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [vocab.encode(word) for word in starting_text.split()]
  for i in range(stop_length):
    input_token = tokens[-1]
    input_token = torch.unsqueeze(torch.unsqueeze(torch.tensor(input_token).type(torch.LongTensor), 0), 0)
    ## if you are using cpu, do not attach input_token to cuda. 
    input_token = input_token.cuda()
    outputs, state = model(input_token, state)
    ## Here we cast outputs to float64 as there are numerical issues at hand
    # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
    last_pred = outputs[-1][-1].type(torch.float64)
    prediciton = nn.functional.softmax(last_pred) 

    ## if you are using cpu, you do not need to change prediciton back to cpu. 
    next_word_idx = sample(prediciton.data.cpu().numpy(), temperature=temp)
    tokens.append(next_word_idx)
    if stop_tokens and vocab.decode(tokens[-1]) in stop_tokens:
      break
  output = [vocab.decode(word_idx) for word_idx in tokens]
  return output

def generate_sentence(model, config, vocab, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(model, config, vocab,  *args, stop_tokens=['<eos>'], **kwargs)


def load_data(debug=False):
  """Loads starter word-vectors and train/dev/test data."""
  vocab = Vocab()
  vocab.construct(get_ptb_dataset('train'))
  encoded_train = np.array(
    [vocab.encode(word) for word in get_ptb_dataset('train')],
    dtype=np.int32)
  encoded_valid = np.array(
    [vocab.encode(word) for word in get_ptb_dataset('valid')],
    dtype=np.int32)
  encoded_test = np.array(
    [vocab.encode(word) for word in get_ptb_dataset('test')],
    dtype=np.int32)
  if debug:
    num_debug = 1024
    encoded_train = encoded_train[:num_debug]
    encoded_valid = encoded_valid[:num_debug]
    encoded_test = encoded_test[:num_debug]
  return encoded_train, encoded_valid, encoded_test, vocab


def compute_loss(ouputs, y, criterion):
  """Compute the loss given the ouput, ground truth y, and the criterion function.

  Hint: criterion should be cross entropy.

  Hint: the input is a list of tensors, each has shape as (batch_size, vocab_size)

  Hint: you need concat the tensors, and reshape its size to (batch_size*num_step, vocab_size).
        Then compute the loss with y. 
  Returns:
    output: A 0-d tensor--averaged loss (scalar)
  """ 
  ### YOUR CODE HERE

  ### END YOUR CODE
  return loss


def run_epoch(our_model, config, model_optimizer, criterion, data, mode='train', verbose=10):
  """
  Run for one epoch. Operations are determined by the mode. 

  """ 
  if mode=='train':
      our_model.zero_grad()
  else:
      our_model.eval()
  ### take the init_hidden as the state input for the 1st step.
  state = our_model.init_hidden()
  total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
  total_loss = []
  for step, (x, y) in enumerate(ptb_iterator(data, config.batch_size, config.num_steps)):
      x = torch.from_numpy(x).type(torch.LongTensor)
      y = torch.from_numpy(y).type(torch.LongTensor)
      ## if you are using cpu, do not attach x,y to cuda. 
      x = x.cuda()
      y = y.cuda()
      outputs, state = our_model(x, state)
      loss = compute_loss(outputs, y, criterion)
      if mode=='train':
        loss.backward()
        model_optimizer.step()
        state= state.detach()
      ## if you are using cpu, you do not need to change loss.data back to cpu. 
      total_loss.append(loss.data.cpu().numpy())
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : pp = {}'.format(step, total_steps, np.exp(np.mean(total_loss))))
        sys.stdout.flush()
  if verbose:
      sys.stdout.write('\r')
  return np.exp(np.mean(total_loss))      
    

def test_RNNLM():
  config = Config()  
  ### load data
  train_data, valid_data, test_data, vocab= load_data(debug=False)
  config.vocab_size= len(vocab)
  ### Initialize the model. If you are using cpu, do not attach model to cuda.  
  our_model = RNNLM_Model(config)
  our_model.cuda()

  ### define the loss (criterion), optimizer
  ### Hint: the criterion should be CE and SGD might be a good choice for optimizer. 

  ### YOUR CODE HERE
  criterion = 
  model_optimizer = 
  ### END YOUR CODE

  best_val_pp = float('inf')
  best_val_epoch = 0
  
  for epoch in range(config.max_epochs):
    print('Epoch {}'.format(epoch))
    start = time.time()
    train_pp = run_epoch(our_model, config, model_optimizer, criterion,train_data,)
    valid_pp = run_epoch(our_model, config, model_optimizer, criterion,valid_data,)
    print('Training perplexity: {}'.format(train_pp))
    print('Validation perplexity: {}'.format(valid_pp))
    if valid_pp < best_val_pp:
      best_val_pp = valid_pp
      best_val_epoch = epoch
      state = {'net': our_model.state_dict()} 
      torch.save(state, './ckpt.pth')
    if epoch - best_val_epoch > config.early_stopping:
      break
    print('Total time: {}'.format(time.time() - start))
    
  ## After training, we load the model and test it.  
  checkpoint = torch.load('./ckpt.pth')
  our_model.load_state_dict(checkpoint['net'])
  test_pp = run_epoch(our_model, config, model_optimizer, criterion,test_data)
  print('=-=' * 5)
  print('Test perplexity: {}'.format(test_pp))
  print('=-=' * 5)
    
  ### Then we generate some sentence..  
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1   
  gen_model = RNNLM_Model(gen_config)
  gen_model.cuda()
  gen_model.load_state_dict(checkpoint['net'])
  starting_text = 'in palo alto'
  while starting_text:
    print(' '.join(generate_sentence(
       gen_model, gen_config,vocab, starting_text=starting_text, temp=1.0)))
    starting_text = input('> ')

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  test_RNNLM()

