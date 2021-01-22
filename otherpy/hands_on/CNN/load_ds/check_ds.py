from otherpy.hands_on.CNN.imports import *


def check_ds_info(info):
    print("> INFO :")
    print(info.splits)
    print(info.splits["train"])
    print(info.splits["test"])
    print("labels :", info.features["label"].names)
    print("total  :", info.features["label"].num_classes)


def check_ds_info_full(info):
    print("> INFO FULL :")
    print(info)


def check_ds_info_return_classes(info):
    print("> INFO - return classes :")
    print(info.splits)
    print(info.splits["train"])
    # print(info.splits["test"])
    print("names    :", info.features["label"].names)
    print("classes  :", info.features["label"].num_classes)
    class_counter = info.features["label"].num_classes
    print("TOTAL train items :", info.splits["train"].num_examples)
    return class_counter


def check_ds_images_shape(data_set, take=3):
    print("[*]Check image shapes - only for first ", take)
    for image, label in data_set.take(take):
        print("> img shape :", image.shape, " & label :", label)


def check_ds_show_images(data_set):
    plt.figure(figsize=(12, 10))
    index = 0
    for image, label in data_set.take(9).cache():
        index += 1
        plt.subplot(3, 3, index)
        # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.imshow(image)
        plt.title("Label: {}".format(label))
        plt.axis("off")
    plt.show()

def check_ds_show_batch_images(data_set):
    plt.figure(figsize=(12, 10))
    index = 0
    for image, label in data_set.take(1):
        for i in range(9):
            index += 1
            plt.subplot(3, 3, index)
            # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            plt.imshow(image[i].numpy().astype("uint8"))
            plt.title("Label: {}".format(label[i]))
            plt.axis("off")
    plt.show()
