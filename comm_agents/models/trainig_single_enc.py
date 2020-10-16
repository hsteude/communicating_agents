import torch
from comm_agents.data.datahandler import RefExpDataset
from comm_agents.models.model_single_enc import SingleEncModel

breakpoint()

# initialize dataset
data = RefExpDataset()

# Initialize model
model = SingleEncModel(observantion_size=data.observations.shape[1],
                       lat_space_size=data.hidden_states.shape[1],
                       question_size=2,
                       num_decoding_agents=4,
                       num_dec_hidden=100, num_enc_hidden=100)

# Define loss and optimizer
learning_rate = 0.01
n_iters = 100

# debug model forward pass
model(data.observations[0], data.questions[0])


loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(n_iters):
    # predict = forward pass with our model
    y_predicted = model(X)

    # loss
    l = loss(Y, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()  # unpack parameters
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
