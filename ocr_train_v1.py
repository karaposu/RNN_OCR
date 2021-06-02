
from tqdm import tqdm
from tqdm import trange, tqdm


def train_fn(model, train_loader, optimizer,num_epochs):
    model.train()
    for epoch in range(1, num_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                # a=data
                optimizer.zero_grad()
                output,loss = model(data,target)
                # output,loss = model(**data)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                # loss = F.nll_loss(output, target)

                loss.backward()
                optimizer.step()


                # sleep(0.1)
    return output,loss


# def train_fn(model, train_loader, optimizer):
#     model.train()
#     for epoch in range(1, 5):
#         with tqdm(train_loader, unit="batch") as tepoch:
#             for data, target in tepoch:
#                 tepoch.set_description(f"Epoch {epoch}")
#
#                 # data, target = data.to(device), target.to(device)
#                 optimizer.zero_grad()
#                 output = model(data)
#                 predictions = output.argmax(dim=1, keepdim=True).squeeze()
#                 loss = F.nll_loss(output, target)
#                 correct = (predictions == target).sum().item()
#                 accuracy = correct / 20
#
#                 loss.backward()
#                 optimizer.step()
#
#                 tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
#                 # sleep(0.1)
#     return None

    # return fin_loss / len(data_loader)


# def train_fn(model, data_loader, optimizer):
#     model.train()
#     fin_loss = 0
#     print("tk begins :")
#     tk0 = trange(len(data_loader), total=len(data_loader))
#     # print("tk0 :", tk0)
#
#     for data in tk0:
#         # print("data len :", len(data))
#         # print("data [0] :", data[0].shape)
#         pass
#
#     return fin_loss / len(data_loader)

# def train_fn(model, data_loader, optimizer):
#     model.train()
#     fin_loss = 0
#     tk0 = tqdm(data_loader, total=len(data_loader))
#     print("tk0 :", tk0)
#
#     for data in tk0:
#         print("data len :", len(data))
#         print("data [0] :", data[0].shape)
#         print("data [1] :", data[1].shape)
#         for key, value in data.items():
#             data[key] = value
#         optimizer.zero_grad()
#         _, loss = model(**data)
#         loss.backward()
#         optimizer.step()
#         fin_loss += loss.item()
#     return fin_loss / len(data_loader)

    # def train_fn(model, data_loader, optimizer):
    # model.train()
    # fin_loss = 0
    # tk0 = tqdm(data_loader, total=len(data_loader))
    # print("tk0 :", tk0)
    #
    # for data in tk0:
    #     print("data len :", len(data))
    #     print("data [0] :", data[0].shape)
    #     print("data [1] :", data[1].shape)
    #     for key, value in data.items():
    #         data[key] = value
    #     optimizer.zero_grad()
    #     _, loss = model(**data)
    #     loss.backward()
    #     optimizer.step()
    #     fin_loss += loss.item()
    # return fin_loss / len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            data[key] = value
        batch_preds, loss = model(**data)
        fin_loss += loss.item()
        fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(data_loader)



def train(model, train_loader, num_epochs, criterion, optimizer):

    # Train the model
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [N, 1, 28, 28]
            # resized: [N, 28, 28]
            if i==1 :
                print(images.shape)
                print(labels.shape)
            # images = images.reshape(-1, sequence_length, input_size)


            # Forward pass
            outputs , loss= model(images)
            # loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')


# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
#
#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')



# if __name__ == "__main__":
#     ocr_train_v1.py()