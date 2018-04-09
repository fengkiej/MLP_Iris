from src.model import Model
import matplotlib.pyplot as plt


def main():
    f = open("iris.dataset")
    dataset = [[(float(x)) for x in line.split(" ")] for line in f]
    dataset = dataset[0:150]
    f.close()

    model = Model(0.2, [4, 10, 3])

    # k-fold cross validation
    k = 5
    fold = []
    fold_len = int(len(dataset) / k)
    # print(dataset)
    for i in range(0, k):
        r = {'validation_data': dataset[0:fold_len], 'training_data': dataset[fold_len:len(dataset)]}
        fold.append(r)
        dataset = shift(dataset, -fold_len)

    print("init weights:", model.layer_weights)

    for j in range(0, k):
        print(fold[j]['training_data'])
        model.train(fold[j]['training_data'], fold[j]['validation_data'], 100)

    print("final weights:", model.layer_weights)

    plt.plot(model.error)
    plt.plot(model.accuracy)
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend(['Error', 'Accuracy'], loc='upper left')
    plt.show()

    #     print("training...")
    #     print("training completed")


def shift(seq, shift=1):
    return seq[-shift:] + seq[:-shift]


if __name__ == '__main__':
    main()
