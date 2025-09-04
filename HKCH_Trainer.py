import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class HKCHTrainer:

    def __init__(self, image_model, text_model, device='cuda'):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.device = device
        
        self.criterion = ImprovedTernaryLoss()
        self.optimizer = optim.Adam(
            list(image_model.parameters()) + list(text_model.parameters()),
            lr=0.001, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
    
    def train_epoch(self, train_loader):
        self.image_model.train()
        self.text_model.train()
        total_loss = 0
        
        for images, texts, labels in tqdm(train_loader, desc="Training"):
            images, texts, labels = images.to(self.device), texts.to(self.device), labels.to(self.device)
            

            hash_img = self.image_model(images)
            hash_txt = self.text_model(texts)
            

            loss = self.criterion(hash_img, hash_txt, labels)
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(train_loader)
    
    def evaluate(self, query_loader, retrieval_loader):
        self.image_model.eval()
        self.text_model.eval()
        
        with torch.no_grad():

            retrieval_imgs, retrieval_txts, retrieval_labs = self.extract_features(retrieval_loader)
            query_imgs, query_txts, query_labs = self.extract_features(query_loader)

            i2t_map = self.calculate_map(query_imgs, retrieval_txts, query_labs, retrieval_labs)
            t2i_map = self.calculate_map(query_txts, retrieval_imgs, query_labs, retrieval_labs)
            
        return i2t_map, t2i_map
    
    def extract_features(self, data_loader):
        pass
    
    def calculate_map(self, query, database, query_labels, db_labels):
        pass