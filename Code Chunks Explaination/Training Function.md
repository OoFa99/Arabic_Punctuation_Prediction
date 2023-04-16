# Training Function #

This is a TensorFlow function for a single training step in an encoder-decoder model.

The function takes three inputs:

- `inp`: an input tensor (batch_size x max_length_inp)
- `targ`: a target tensor (batch_size x max_length_targ)
- `enc_hidden`: an initial hidden state tensor for the encoder (batch_size x units)

The function first initializes `loss` and `accuracy` variables to zero. Then, it uses a `tf.GradientTape()` context to record the operations that are executed on the input and target tensors.

Inside the `tf.GradientTape()` context, the function passes the `inp` and `enc_hidden` tensors to the `encoder` and gets `enc_output` and `enc_hidden` as output. The `enc_output` tensor is then passed to the `decoder`, along with the initial `dec_hidden` state, which is set to `enc_hidden`.

The `decoder` then produces `predictions`, a tensor of shape `(batch_size, vocab_size)`, which contains the predicted probabilities for each word in the vocabulary. The function uses `predictions` and the `targ` tensor to compute the `loss` by calling the `loss_function`. The `loss` is accumulated over each time step of the decoder output.

The function then computes the gradients of the `loss` with respect to the trainable variables of the `encoder` and `decoder`, which are concatenated into a single list. The gradients are then applied to the variables using the `optimizer`.

Finally, the function returns the `batch_loss` (i.e., the average loss per target word in the batch).



### Calling the function

This code trains the sequence-to-sequence model for a specified number of epochs, where each epoch refers to one pass over the entire training dataset. The training process involves the following steps:

1. Initializing the hidden state of the encoder to a zero vector.
2. Iterating over the batches of the training dataset using the `train_dataset` object obtained earlier. For each batch, the `train_step` function is called with the current input sequence, target sequence, and encoder hidden state as inputs.
3. The `train_step` function computes the loss and gradients for the current batch using the forward pass through the encoder and decoder, and applies the gradients to update the model parameters using the Adam optimizer.
4. The function returns the batch loss which is then accumulated to compute the total loss for the epoch.
5. The code prints the current epoch number, batch number, and the batch loss after every 100 batches.
6. After every two epochs, the model and optimizer parameters are saved to the specified directory using the `checkpoint` object.
7. After each epoch, the average loss over all batches is printed along with the time taken for that epoch.

The `EPOCHS` variable specifies the number of epochs to train the model for. The outer loop iterates over this range, and for each epoch, the code initializes the encoder hidden state to a zero vector, and accumulates the loss over all batches. At the end of the epoch, the average loss is computed and printed along with the time taken for that epoch. After every two epochs, the model is saved to the specified checkpoint directory.