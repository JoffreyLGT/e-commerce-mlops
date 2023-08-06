import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.sparse import SparseTensor


def load_data(datadir: str = "data") -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(f"{datadir}/X.csv", index_col=0),
            pd.read_csv(f"{datadir}/y.csv", index_col=0),
        ],
        axis=1,
    )


def get_img_name(productid: int, imageid: int) -> str:
    """Return the filename of the image.

    Arguments:
    - productid: int - "productid" field from the original DataFrame.
    - imageid: int - "imageid" field from the original DataFrame.

    Return:
    A string containing the filename of the image. Example: image_1000076039_product_580161.jpg
    """
    return f"image_{imageid}_product_{productid}.jpg"


def get_imgs_filenames(
    productids: list[int], imageids: list[int], folder: str | None = None
) -> list[str]:
    """Return a list of filenames from productids and imagesids.

    Arguments:
    - productids: list of product ids
    - imageids: list of image ids
    - folder: folder containing the images. Used only to return a full path.

    Return:
    A list of the same size as productids and imageids containing the filenames.
    """
    if len(productids) != len(imageids):
        raise ValueError("productids and imageids should be the same size")
    if folder is None:
        return [
            get_img_name(productid, imageid)
            for productid, imageid in zip(productids, imageids)
        ]
    else:
        return [
            os.path.join(folder, get_img_name(productid, imageid))
            for productid, imageid in zip(productids, imageids)
        ]


def remove_white_stripes(img_array: np.ndarray) -> np.ndarray:
    """Remove outer white lines and columns.

    Arguments:
        img_array: imaged loaded into a np.ndarray.

    Returns:
        The same array without the outer white stripes.

    Example:
        remove_white_stripes(np.asarray(Image.open("my_image.png")))
    """
    top_line = -1
    right_line = -1
    bottom_line = -1
    left_line = -1

    i = 1
    while top_line == -1 or bottom_line == -1 or left_line == -1 or right_line == -1:
        if top_line == -1 and img_array[:i].mean() != 255:
            top_line = i
        if bottom_line == -1 and img_array[-i:].mean() != 255:
            bottom_line = i
        if left_line == -1 and img_array[:, :i].mean() != 255:
            left_line = i
        if right_line == -1 and img_array[:, -i:].mean() != 255:
            right_line = i

        i += 1
        if i >= img_array.shape[0]:
            break

    if top_line == -1 or bottom_line == -1 or left_line == -1 or right_line == -1:
        return img_array
    else:
        return img_array[top_line:-bottom_line, left_line:-right_line, :]


def crop_resize_img(
    filename: str,
    imput_img_dir: str,
    output_img_dir: str,
    width: int,
    height: int,
    keep_ratio: bool,
    grayscale: bool = False,
) -> None:
    """Crop, resize and apply a grayscale filter to the image.

    Arguments:
    - filename - str: name of the image to process. Must contain the extension.
    - input_img_dir - str: directory containing the image.
    - output_img_dir - str: directory to save the processed image in.
    - width, height - int: width and height of the processed image.
    - keep_ratio - bool: True to keep the image ratio and eventualy add some white stripes around to fill empty space. False to stretch the image.
    - grayscale - bool: True to remove the colors and set them as grayscale.
    """
    # Remove the outer white stripes from the image
    img_array = np.asarray(Image.open(imput_img_dir + filename))
    new_img_array = remove_white_stripes(img_array)
    new_img = Image.fromarray(new_img_array)

    if keep_ratio:
        new_width = new_img.width
        new_height = new_img.height

        ratio = new_width - new_height
        padding_value = np.abs(ratio) // 2
        padding = ()
        if ratio > 0:
            padding = (0, padding_value, 0, padding_value)
        else:
            padding = (padding_value, 0, padding_value, 0)

        new_img = ImageOps.expand(new_img, padding, (255, 255, 255))

    new_img = new_img.resize((width, height))

    if grayscale:
        new_img = ImageOps.grayscale(new_img)

    new_img.save(f"{output_img_dir}/{filename}")


def get_output_dir(
    width: int, height: int, keep_ratio: bool, grayscale: bool, type: str
):
    result = f"cropped_w{width}_h{height}"
    if keep_ratio:
        result += "_ratio"
    else:
        result += "_stretched"
    if grayscale:
        result += "_graycaled"
    else:
        result += "_colors"

    return os.path.join("data", "images", result, type)


def get_img_full_path(
    width: int,
    height: int,
    keep_ratio: bool,
    grayscale: bool,
    type: str,
    prdtypecode: int,
    filename: str,
):
    output_dir = get_output_dir(width, height, keep_ratio, grayscale, type)
    return os.path.join("data", "images", type, output_dir, prdtypecode, filename)


def convert_sparse_matrix_to_sparse_tensor(X) -> SparseTensor:
    """Convert sparse matrix to sparce tensor.

    Arguments:
    - X: sparse matrix to convert.

    Returns:
    - X matrix converted to sparse tensor.
    """
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))


PRDTYPECODE_DIC = {
    10: 0,
    1140: 1,
    1160: 2,
    1180: 3,
    1280: 4,
    1281: 5,
    1300: 6,
    1301: 7,
    1302: 8,
    1320: 9,
    1560: 10,
    1920: 11,
    1940: 12,
    2060: 13,
    2220: 14,
    2280: 15,
    2403: 16,
    2462: 17,
    2522: 18,
    2582: 19,
    2583: 20,
    2585: 21,
    2705: 22,
    2905: 23,
    40: 24,
    50: 25,
    60: 26,
}


def to_simplified_prdtypecode(y: np.array):
    """Convert the prdtypecode into a simplified equivalent ranging from 0 to 26.

    Arguments:
    - y: list of prdtypecode to convert to a simplified range.

    Returns:
    - y with converted prdtypecode.
    """
    return np.array([PRDTYPECODE_DIC[i] for i in y])


def to_normal_prdtypecode(y: np.array):
    """Convert back a simplified prdtypecode (ranging from 0 to 26) to the original prdtypecode.

    Arguments:
    - y: list of prdtypecode to convert to a the original value.

    Returns:
    - y with original prdtypecode.
    """
    return np.array(
        [
            list(PRDTYPECODE_DIC.keys())[list(PRDTYPECODE_DIC.values()).index(i)]
            for i in y
        ]
    )


def get_model_prediction(y_pred):
    """Get normal prdtypecode from a model prediction returning the probabilities of each prdtypecode.

    Arguments:
    - y: list of predictions for each prdtypecode.

    Returns:
    - a list containing the original prdtypecode with the highest probability.
    """
    list_decision = []
    for y in y_pred:
        list_decision.append(np.argmax(y))
    return np.array(to_normal_prdtypecode(list_decision))


def open_resize_img(filename: str, y) -> None:
    """Open image using the filename and return a resized version of it ready for the image model.

    Argument:
    - filename: complete path to image file including the extension.

    Return:
    - Image matrix in a tensor.
    """
    img = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img, channels=3)
    return (tf.image.resize(img, [224, 224]), y)


CATEGORIES_DIC = {
    10: "Livre",
    1140: "Figurine et produits dérivés",
    1160: "Carte à collectionner",
    1180: "Univers fantastiques",
    1280: "Jouet pour enfant",
    1281: "Jeu de société",
    1300: "Miniature de collection",
    1301: "Loisir",
    1302: "Activité d'extérieur",
    1320: "Accessoire bébé",
    1560: "Meuble d'intérieur",
    1920: "Litterie, rideaux",
    1940: "Epicerie",
    2060: "Décoration d'intérieur",
    2220: "Accessoire animaux de compagnie",
    2280: "Magazine et BD",
    2403: "Livres anciens",
    2462: "Jeu vidéo - Pack",
    2522: "Fourniture de bureau",
    2582: "Meubles extérieur",
    2583: "Piscine",
    2585: "Bricolage",
    2705: "Livre",
    2905: "Jeu vidéo - Jeu",
    40: "Jeu vidéo - Jeu",
    50: "Jeu vidéo - Accessoire",
    60: "Jeu vidéo - Console",
}

DATA_SAMPLE = [
    [
        "Christmas decoration",
        "Décor De Noël Ornement Led Lumières Cerfs Panier Lumineux Maison En Bois Fenêtre Mall",
        "Décor De Noël Ornement Led Lumières Cerfs Panier Lumineux Maison En Bois Fenêtre Mall",
        "assets/image_1257660878_product_3881422405.jpg",
    ],
    [
        "English manual",
        "Exploring English Level 2 Teacher's Resource Manual",
        "",
        "assets/image_859609154_product_93575721.jpg",
    ],
    [
        "Start Wars figurine",
        "Figurine - Star Wars - Stormtrooper - 30th",
        "",
        "assets/image_835089072_product_68594029.jpg",
    ],
    [
        "Pool",
        "Intex - 57495fr - Jeu D'eau Et De Plage - Piscine Carré - Hublot",
        "Longeur: 549 cm Largeur: 30 cm Descriptif produit: Piscine carrée hublot 229x229x56cm.Larges parois fenêtres transparentes sur les ctés.Capacité : 1215 litres environ Nécessite des piles: Non Modèle : 57495E",
        "assets/image_898066724_product_144402962.jpg",
    ],
    [
        "Console NES",
        "Nintendo Nes 2 Manettes Et Mario Bros",
        "",
        "assets/image_1124990133_product_2086705884.jpg",
    ],
]


def get_random_product(prdtypecode, data):
    """Return a random product with prdtypecode from data.

    Arguments:
    - prdtypecode: type of the product to return
    - data: dataframe to use.

    Returns:
    - tuple with (designation, description, image_path)
    """
    product = data[data["prdtypecode"] == prdtypecode].sample(1).iloc[0]
    image_path = get_imgs_filenames(
        productids=[product["productid"]],
        imageids=[product["imageid"]],
        folder="../data/images/image_train",
    )[0]
    return (product["designation"], product["description"], image_path)
