# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from torch import nn
from typing import Literal, Sequence, Union
import torch
import numpy as np


class EvalTool(ABC):
    @abstractmethod
    def score(self, *args, **kwargs):
        pass


class Fidelity(EvalTool):
    def __init__(self, model: nn.Module, steps: int = 40, perturb_method: Literal["mean"] = "mean", device=None):
        self.id = "fidelity"
        self.model = model
        self.steps = steps
        self.perturb_method = perturb_method
        
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        
    def score(self, batch: torch.Tensor, xai_batch: torch.Tensor, target_logits: Union[int, Sequence[int]]):
        if isinstance(target_logits, int):
            target_logits = [target_logits] * len(batch)
        elif len(target_logits) != len(batch):
            raise ValueError("target_logits must be either a single integer or a list of the same length as the batch")

        scores = [
            self._compute_auc(image, xai_map, target_logit)
            for image, xai_map, target_logit in zip(batch, xai_batch, target_logits)
        ]
        return scores

    def _compute_auc(self, image, xai_map, target_logit):
        out = get_MIF_LIF(
            model=self.model,
            target_idx=target_logit,
            img=image,
            saliency_map=xai_map.squeeze(),
            steps=self.steps,
            method=self.perturb_method,
            device=self.device
        )
        ## FIDELITY COMPUTATION SHOULD BE OPTIMIZED !!
        #diff = torch.tensor(out["LIF"]) - torch.tensor(out["MIF"])
        #auc = torch.trapz(diff, torch.linspace(0, 1, self.steps + 1))
        diff = [i-j for i,j in zip(out["LIF"], out["MIF"])]
        auc = np.trapz(diff, np.linspace(0, 1, self.steps+1)) # It should be possible to make in torch.
        return auc.item()


class Compactness(EvalTool):
    def __init__(self):
        self.id = "compactness"

    def score(self, xai_batch: torch.Tensor):
        return [self._compute_compactness(xai_map) for xai_map in xai_batch]

    def _compute_compactness(self, xai_map):
        map_abs = torch.abs(xai_map)
        return (map_abs <= 0.05 * torch.max(map_abs)).sum().item() / map_abs.numel()


import numpy as np
from PIL import ImageFilter
import PIL
import torch
import matplotlib.pyplot as plt


# Definitions for MIF/LIF perturbation
def get_grid2d(arr, descending=True):
    """
    The function `get_grid2d` takes a 2D array and returns a sorted 2D grid with values from the input
    array along with their corresponding x and y indices, allowing for sorting in descending order if
    specified.
    
    :param arr: A 2D numpy array that you pass into the `get_grid2d` function
    :param descending: A boolean parameter
    that determines whether the output grid should be sorted in descending order based on the values of
    the input array. When `descending=True`, the grid will be sorted in descending order, and when
    `descending=False`, the, defaults to True (optional)
    :return: The function `get_grid2d` returns a 2D grid where the first column contains the values of
    the input array, and the second and third columns contain the x and y indices of the input array.
    The grid is sorted based on the values of the input array, with the option to sort in descending
    order if the `descending` parameter is set to `True`.
    """
    grid = []
    xs, ys = arr.shape
    x_indices, y_indices = np.meshgrid(np.arange(xs), np.arange(ys), indexing="ij")
    grid = np.stack((arr, x_indices, y_indices), axis=-1).reshape(-1, 3)
    grid = np.array(grid)
    grid_sorted = grid[grid[:, 0].argsort()]
    if descending:
        return grid_sorted[::-1]
    else:
        return grid_sorted
    
def perturb_n(img, attribution, n, order="MIF", method="blur"):
    """
    The function `perturb_n` perturbs an image by either blurring or setting to 0 the n most important
    or least important pixels based on an attribution map.
    
    :param img: The `img` parameter is a PIL image that represents the image you want to perturb. It is
    the input image that will be modified based on the attribution map and the specified perturbation
    method
    :param attribution: A 2D numpy array
    representing the attribution map. This attribution map contains information about the importance of
    each pixel in the image for a given task or model. The values in the array indicate the level of
    importance assigned to each
    :param n: The number of pixels to perturb
    in the image. This value determines how many pixels will be modified based on their importance
    scores in the attribution map
    :param order: Whether to perturb the
    most important pixels first ("MIF"; default) or the least important pixels first ("LIF") based on the
    attribution map. (optional)
    :param method: A string to specify the perturbation method.
    There are three options for the `method` parameter: blur (default), const, mean.
    :return: The function `perturb_n` returns two values: the perturbed image (`img_cp`) and the sum of
    the importance scores of the perturbed pixels (`attribution_sum`).
    """
    if order == "MIF":
        descending = True
    else:
        descending = False
    grid = get_grid2d(np.array(attribution), descending=descending)

    

    x = grid[:, 1].astype(np.int64)[:n]
    y = grid[:, 2].astype(np.int64)[:n]

    img_cp = np.copy(np.array(img.cpu()))
    # make sure that the image is in the format (C, H, W)
    if img_cp.shape[-1] <= 3:
        img_cp = np.moveaxis(img_cp, 0, -1)

    for c in range(img_cp.shape[0]):
        if method == "blur":
            blurred_image = np.array(img.filter(ImageFilter.GaussianBlur(radius=14)))
            # assumes that the PIL image is in RGB format
            img_cp[c, x, y] = blurred_image[x, y, c]
        elif method == "const":
            img_cp[c, x, y] = 0
        elif method == "mean":
            img_cp[c, x, y] = np.mean(img_cp[c])
        else:
            raise ValueError("Invalid method")
    # plt.matshow(img_cp[0])
    # plt.show()
    # print(len(x), len(y))
    attribution_sum = np.sum(np.array(attribution[x, y]))
    return img_cp, attribution_sum

def get_MIF_LIF(model, target_idx, img, saliency_map, steps, method, device, transform=None):
    """
    The function `get_MIF_LIF` perturbs an image in steps based on saliency map importance scores in
    both most important first (MIF) and least important first (LIF) order, recording model predictions
    and sum of importance scores at each step.
    
    :param model: A PyTorch model that will be used for making predictions on
    the perturbed images during the MIF (Most Important First) and LIF (Least Important First)
    perturbation process. This model should be capable of processing input images and providing
    predictions for different classes
    :param target_idx: The index of
    the target class for which you want to analyze the perturbation effects. This index corresponds to
    the specific class in the model's output for which you are interested in understanding the impact of
    perturbations
    :param img: A PIL image, a torch tensor
    (CxHxW), or a numpy array (CxHxW). It represents the original image that will be perturbed.
    :param saliency_map: A 2D
    numpy array that contains the saliency map associated with the input image. The saliency map
    highlights the important regions or features in the image that contribute the most to the model's
    :param steps: The number of steps to
    perturb the image in the most important first (MIF) and least important first (LIF) order. It
    determines how many iterations of perturbation will be performed on the image to analyze
    :param method: The perturbation method to be applied during the image perturbation process.
    It can take on two possible values:
    :param transform: A transformation function that preprocesses the images.
    It is a function that applies transformations to the perturbed images to match the preprocessing
    requirements of the model.
    :return: A dictionary containing the following keys and their corresponding values:
    - "MIF":
    - "sum_MIF": 
    - "LIF":
    - "sum_LIF":
    """

    res_MIF = []
    res_LIF = []
    sum_MIF = []
    sum_LIF = []
    if isinstance(img, PIL.Image.Image):
        img_size = img.size
    elif isinstance(img, torch.Tensor) or isinstance(img, np.ndarray):
        if method == "blur":
            raise ValueError("Method 'blur' is not supported for torch tensor or numpy array input.")
        img_size = img.shape[-2:]
        # check if image dimensions were as expected (assuming image is RGB or grayscale)
        t = [s <= 3 for s in img_size]
        if True in t:
            raise ValueError("Image dimensions were not as expected, should be (C,H,W).")
    else:
        raise ValueError("Invalid image type, must be PIL image, torch tensor or numpy array.")
    
    for k in range(steps + 1):
        num_remove = int(k / steps * img_size[0] * img_size[1])
        perturbed_MIF, sum_MIF_k = perturb_n(
            img, saliency_map, num_remove, order="MIF", method=method
        )
        perturbed_LIF, sum_LIF_k = perturb_n(
            img, saliency_map, num_remove, order="LIF", method=method
        )
        if transform:
            perturbed_MIF = transform(perturbed_MIF).to(device)
            perturbed_LIF = transform(perturbed_LIF).to(device)

        if not isinstance(perturbed_MIF, torch.Tensor):
            perturbed_MIF = torch.tensor(perturbed_MIF).to(device)
        if not isinstance(perturbed_LIF, torch.Tensor):
            perturbed_LIF = torch.tensor(perturbed_LIF).to(device)

        with torch.no_grad():
            pred_MIF = model(perturbed_MIF.unsqueeze(0))
            pred_LIF = model(perturbed_LIF.unsqueeze(0))
        res_MIF.append(pred_MIF[0, target_idx].cpu().numpy())
        res_LIF.append(pred_LIF[0, target_idx].cpu().numpy())
        sum_MIF.append(sum_MIF_k)
        sum_LIF.append(sum_LIF_k)
    return {"MIF": res_MIF, "sum_MIF": sum_MIF, "LIF": res_LIF, "sum_LIF": sum_LIF}

# Definitions for box sensitivity
def perturb_box(img, x, y, h, method="blur"):
    """
    The function perturbs a box of pixels in an image by either blurring or setting them to 0 based on
    the specified method.
    
    :param img: The `img` parameter is a PIL image
    :param x: The x-coordinate of the top
    left corner of the box that you want to perturb in the image
    :param y: The y-coordinate of the top
    left corner of the box that you want to perturb in the image.
    :param h: The height and width of the square box that will be perturbed in the image.
    :param method: The perturbation
    method to be applied to the box in the image. It can take two values: blur (default) or const.
    :return: The function `perturb_box` returns the perturbed image after applying the specified
    perturbation method to a square of pixels in the input image.
    """
    img_cp = np.copy(np.array(img))
    blurred_image = np.array(img.filter(ImageFilter.GaussianBlur(radius=14)))
    for c in range(3):
        if method == "blur":
            img_cp[x:x+h, y:y+h, c] = blurred_image[x:x+h, y:y+h, c]
        elif method == "const":
            img_cp[x:x+h, y:y+h, c] = 0
        else:
            raise ValueError("Invalid method")
    return img_cp

def box_sensitivity(saliency_map, original_img, transform, model, target_idx, box_size, method, num_boxes=100):
    """
    The function `box_sensitivity` calculates sensitivity-n by perturbing pixels inside randomly placed
    boxes and records the sum of importance scores of perturbed pixels and the change in model output
    for the target class. Similar to sensitivity-n (https://arxiv.org/pdf/1711.06104) but with box.
    
    :param saliency_map: A 2D numpy array representing the importance
    scores of each pixel in the image. These scores indicate how important each pixel is for the model's
    prediction
    :param original_img: A PIL image
    representing the original image that you want to perturb. This image will be used to generate
    perturbed versions by applying perturbations inside randomly placed boxes
    :param transform: A preprocessing
    function that prepares the input image in a way that matches the requirements of the model. This
    typically involves resizing, normalization, and any other necessary transformations to ensure the
    input image is in the correct format for the model to make
    :param model: A PyTorch model
    that you are using for your task. This model is typically a neural network that you have trained for
    a specific task, such as image classification. The model takes input data, processes it through its
    layers
    :param target_idx: The index
    of the target class for which you want to calculate the sensitivity. This index corresponds to the
    specific class in the model's output for which you are interested in analyzing the sensitivity to
    perturbations within the image
    :param box_size: The size of
    the box that will be used to perturb pixels inside the image. It determines the dimensions of the
    square box that will be randomly placed on the image for perturbation
    :param method: The perturbation
    method used for modifying the image inside the randomly placed boxes. It can take on two possible
    values:
    :param num_boxes: How many
    times the image will be perturbed by placing randomly located boxes on it. This parameter determines
    the number of perturbations that will be applied to the image to calculate the sensitivity-n metric,
    defaults to 100 (optional)
    :return: Two lists: `sum_relevance` and `delta_out`.
    """
    perturbed = []
    coordinates = []

    batch_size = 50

    img_size = original_img.size

    x_range = np.random.randint(0, img_size[0] - box_size, size=num_boxes)
    y_range = np.random.randint(0, img_size[1] - box_size, size=num_boxes)

    for x_pos, y_pos in zip(x_range, y_range):
            img_perturbed = perturb_box(original_img, x_pos, y_pos, box_size, method)
            img_perturbed = transform(img_perturbed)
            coordinates.append({"x": x_pos, "y": y_pos})

            perturbed.append(img_perturbed)

    num_batches = int(num_boxes / batch_size)

    perturb_out = []
    with torch.no_grad():
        input_img = transform(original_img).unsqueeze(0).cuda()
        out_original = model(input_img)
    for i in range(num_batches):
        batch = perturbed[i*batch_size:(i+1)*batch_size]
        batch = torch.stack(batch).cuda()
        with torch.no_grad():
            out = model(batch)
            perturb_out.append(out)
    if num_boxes % batch_size != 0:
        batch = perturbed[num_batches*batch_size:]
        batch = torch.stack(batch).cuda()
        with torch.no_grad():
            out = model(batch)
            perturb_out.append(out)

    perturb_out = torch.cat(perturb_out)

    sum_relevance = [torch.sum(saliency_map[coordinates[i]["x"]:coordinates[i]["x"]+box_size,coordinates[i]["y"]:coordinates[i]["y"]+box_size]) for i in range(len(coordinates))]
    delta_out = [ (out_original[0][target_idx] - perturb_out[i][target_idx]).cpu().detach().numpy() for i in range(len(perturb_out))]
    
    return sum_relevance, delta_out

# Definition of compactness
def compactness(saliency_map, threshold=0.05):
    """
    The `compactness` function calculates the ratio of saliency map values within a threshold of the
    maximum absolute value.
    
    :param saliency_map: A tensor that contains the importance scores (e.g., saliency map) for
    different parts of an input data (e.g., an image) with respect
    to a specific task.
    :param threshold: The proportion 
    of the maximum absolute value in the saliency map that is used as the threshold for determining the
    compactness ratio. By default, the threshold is set to 5% (0.05)
    :return: The `compactness` function returns the ratio of the number of elements in the saliency map
    that are less than or equal to the threshold of the maximum absolute value in the map to the total number of
    elements in the saliency map.
    """
    map_abs = torch.abs(saliency_map)
    return len(map_abs[map_abs <= threshold * torch.max(map_abs)])/len(map_abs.flatten())

