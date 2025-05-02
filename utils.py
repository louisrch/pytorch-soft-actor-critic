import math
import torch
# model imports
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
import numpy as np
from PIL import Image

QUERIES = {
    "CartPole-v1" : "What is in this picture ? The goal of the agent is to keep the pole upright. Is the pole upright in this picture ? If not, edit this picture, while preserving the proportions, such that the pole is upright. If it is already upright, do not do anything." 
}

device = "cpu"
model, preprocess = clip.load("ViT-B/32",device=device)


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def get_goal_embedding(env_name, env):
    goal = None
    if env_name == "CartPole-v1":
        goal_image = env.render()
    else:
        return None
    embedding = get_image_embedding(goal_image)
    return embedding

def get_current_state_embedding(env):
    image = env.render()
    return get_image_embedding(image)


def get_image_embedding(image, model = model):
    """
    encodes the image using the model
    image : input image
    model : encoding model
    credit : https://github.com/openai/CLIP
    """

    image_input = Image.fromarray(image)
    with torch.no_grad():
        features = model.encode_image(preprocess(image_input).unsqueeze(0).to(device))
    return features
    
