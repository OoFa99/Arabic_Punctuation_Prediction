# Prediction Function #

### Evaluate function ###

This code defines a function `evaluate` which takes in a sentence as input and returns the predicted result, the original input sentence, and the attention weights.

First, the function preprocesses the input sentence by adding start and end tokens and tokenizing it using the input tokenizer. The tokenized sequence is then padded to the `input_max_length`.

Then, the function initializes the encoder's hidden state with zeros and passes the input sequence to the encoder to get the encoded output and hidden state.

Next, the decoder's hidden state is set to the encoder's hidden state and the decoder's input is initialized with the start token.

The function then enters a loop where it predicts the next word in the sequence by passing the decoder's input, decoder's hidden state, and encoded output to the decoder. The predicted word is obtained by taking the argmax of the predicted probability distribution over the target vocabulary.

The predicted word is then appended to the `result` string. If the predicted word is the end token, the function returns the `result` string, the original input sentence, and the attention weights.

If the predicted word is not the end token, the predicted word's index is passed to the decoder as input in the next iteration, and the loop continues until either the end token is predicted or the maximum target sequence length is reached. The attention weights for each time step are also stored in the `attention_plot` array.

Finally, the function returns the `result` string, the original input sentence, and the `attention_plot` array.



### Predict function ###

The `predict()` function takes an input sentence as an argument and uses the `evaluate()` function to predict the output sentence with appropriate punctuations. The function first calls the `evaluate()` function to get the predicted result, the input sentence, and the attention plot.

The function then creates an empty string variable `predict` to store the predicted sentence. The `sentence_list` variable is created by splitting the input sentence into a list of words. The `<start>` token is removed from this list using the `pop()` method.

The `result_list` variable is created by splitting the predicted result into a list of words. The function then iterates over each word in `sentence_list` and checks if the corresponding word in `result_list` is equal to the string `"space"`. If it is, then the function adds a space character to `predict`. Otherwise, the function concatenates the current word in `sentence_list` with the corresponding word in `result_list` and adds it to `predict`.

Finally, the function prints the input sentence and the predicted punctuation sentence using the `print()` function. The `plot_attention()` function is then called to visualize the attention weights. The attention plot is created by selecting the part of the attention weights corresponding to the length of the predicted sentence and the length of the input sentence.