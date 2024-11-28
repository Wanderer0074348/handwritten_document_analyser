import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PaperProcessor import MathPaperProcessor
class HandwritingFeatureExtractor(nn.Module):
    def __init__(self, num_classes=128):
        super(HandwritingFeatureExtractor, self).__init__()
        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Attention Module
        self.attention = nn.MultiheadAttention(128, 8)
        
        # Feature Embedding Layer
        self.fc = nn.Linear(128, num_classes)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        try:
            # CNN Feature Extraction
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            
            # Reshape for attention
            batch_size, c, h, w = x.size()
            if h * w == 0:
                # Handle case where dimensions become too small
                x = x.view(batch_size, c, -1)
                x = x.mean(dim=2)  # Global average pooling
                return self.fc(x)
            
            x = x.view(batch_size, c, -1).permute(2, 0, 1)
            
            # Self-attention
            attn_output, _ = self.attention(x, x, x)
            
            # Global average pooling
            x = attn_output.mean(dim=0)
            
            # Final embedding
            x = self.dropout(x)
            x = self.fc(x)
            return x
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            # Return zero tensor with correct dimensions
            return torch.zeros(1, num_classes, device=x.device)

class DocumentSimilarityLearner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HandwritingFeatureExtractor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.document_embeddings = {}
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def preprocess_section(self, section):
        try:
            if isinstance(section, dict):
                if 'text' in section:
                    from PIL import Image, ImageDraw
                    img = Image.new('L', (224, 224), color='white')
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10), str(section['text']), fill='black')
                    return self.transform(img).to(self.device)
                elif 'geometry' in section:
                    coords = torch.tensor(section['geometry'], dtype=torch.float32)
                    coords = coords.view(1, 1, -1)
                    return coords.to(self.device)
            return None
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return None

    def train_on_document(self, doc_id, sections):
        self.model.train()
        document_embeddings = []
        
        print(f"Processing document {doc_id} with {len(sections)} sections")
        
        for idx, section in enumerate(sections):
            try:
                tensor_data = self.preprocess_section(section)
                if tensor_data is not None:
                    if tensor_data.dim() == 3:
                        tensor_data = tensor_data.unsqueeze(0)
                    
                    self.optimizer.zero_grad()
                    features = self.model(tensor_data)
                    document_embeddings.append(features.detach().cpu())
                    print(f"Successfully processed section {idx}")
            except Exception as e:
                print(f"Error processing section {idx} in document {doc_id}: {str(e)}")
                continue
        
        if document_embeddings:
            self.document_embeddings[doc_id] = document_embeddings
            print(f"Successfully stored embeddings for document {doc_id}")
        else:
            print(f"No embeddings generated for document {doc_id}")

    def compare_documents(self, doc_id1, doc_id2):
        """Compare two documents and return similarity scores"""
        if doc_id1 not in self.document_embeddings or doc_id2 not in self.document_embeddings:
            raise ValueError(f"Missing embeddings for documents {doc_id1} and/or {doc_id2}")

        embeddings1 = self.document_embeddings[doc_id1]
        embeddings2 = self.document_embeddings[doc_id2]
        
        similarity_scores = []
        
        for i, emb1 in enumerate(embeddings1):
            section_scores = []
            for j, emb2 in enumerate(embeddings2):
                try:
                    # Convert to CPU tensors
                    emb1_cpu = emb1.cpu() if torch.is_tensor(emb1) else torch.tensor(emb1)
                    emb2_cpu = emb2.cpu() if torch.is_tensor(emb2) else torch.tensor(emb2)
                    
                    # Ensure proper dimensions
                    if emb1_cpu.dim() == 1:
                        emb1_cpu = emb1_cpu.unsqueeze(0)
                    if emb2_cpu.dim() == 1:
                        emb2_cpu = emb2_cpu.unsqueeze(0)
                    
                    # Calculate cosine similarity
                    similarity = F.cosine_similarity(emb1_cpu, emb2_cpu, dim=1)
                    score = similarity.mean().item()
                    
                    section_scores.append({
                        'section1_idx': i,
                        'section2_idx': j,
                        'similarity': score
                    })
                except Exception as e:
                    print(f"Error comparing sections {i} and {j}: {str(e)}")
                    section_scores.append({
                        'section1_idx': i,
                        'section2_idx': j,
                        'similarity': 0.0
                    })
            
            similarity_scores.append(section_scores)
        
        return similarity_scores

    def get_embedding_stats(self):
        """Debug method to print embedding statistics"""
        for doc_id, embeddings in self.document_embeddings.items():
            print(f"\nDocument {doc_id} statistics:")
            print(f"Number of sections: {len(embeddings)}")
            for i, emb in enumerate(embeddings):
                print(f"Section {i} shape: {emb.shape}")

class DocumentProcessor:
    def __init__(self):
        self.math_processor = MathPaperProcessor()
        self.similarity_learner = DocumentSimilarityLearner()
    
    def process_and_compare_documents(self, doc_paths):
        document_sections = {}
        
        # Process each document
        for doc_id, path in enumerate(doc_paths):
            try:
                print(f"\nProcessing document {doc_id}: {path}")
                results = self.math_processor.process_paper(path)
                sections = []
                
                # Process each region type
                for region_type in ['header', 'question', 'solution_text', 'equations']:
                    if region_type in results:
                        for item in results[region_type]:
                            if isinstance(item, dict):
                                # Include region type in section data
                                item['region_type'] = region_type
                                sections.append(item)
                
                if sections:
                    document_sections[doc_id] = sections
                    print(f"Found {len(sections)} sections in document {doc_id}")
                    self.similarity_learner.train_on_document(doc_id, sections)
                else:
                    print(f"No sections found in document {doc_id}")
                
            except Exception as e:
                print(f"Error processing document {doc_id}: {str(e)}")
                continue
        
        # Compare documents
        comparisons = {}
        for i in range(len(doc_paths)):
            for j in range(i + 1, len(doc_paths)):
                try:
                    if i in self.similarity_learner.document_embeddings and j in self.similarity_learner.document_embeddings:
                        similarity_scores = self.similarity_learner.compare_documents(i, j)
                        comparisons[(i, j)] = similarity_scores
                        print(f"Successfully compared documents {i} and {j}")
                    else:
                        print(f"Missing embeddings for documents {i} and/or {j}")
                except Exception as e:
                    print(f"Error comparing documents {i} and {j}: {str(e)}")
                    continue
        
        return comparisons
class DocumentProcessor:
    def __init__(self):
        self.math_processor = MathPaperProcessor()
        self.similarity_learner = DocumentSimilarityLearner()
        
    def preprocess_section(self, section_image):
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return transform(section_image).unsqueeze(0)
        
    def process_and_compare_documents(self, doc_paths):
        """Process multiple documents and compare their similarities"""
        document_sections = {}
        
        # Process each document
        for doc_id, path in enumerate(doc_paths):
            try:
                results = self.math_processor.process_paper(path)
                sections = []
                
                # Extract sections with their content
                for region_type in ['header', 'question', 'solution_text', 'equations']:
                    if region_type in results:
                        for item in results[region_type]:
                            if isinstance(item, dict):
                                sections.append({
                                    'type': region_type,
                                    'content': item
                                })
                
                document_sections[doc_id] = sections
                self.similarity_learner.train_on_document(doc_id, sections)
                print(f"Successfully processed document {doc_id}")
                
            except Exception as e:
                print(f"Error processing document {doc_id}: {str(e)}")
                continue
        
        # Compare all document pairs
        comparisons = {}
        for i in range(len(doc_paths)):
            for j in range(i + 1, len(doc_paths)):
                try:
                    similarity_scores = self.similarity_learner.compare_documents(i, j)
                    comparisons[(i, j)] = similarity_scores
                except Exception as e:
                    print(f"Error comparing documents {i} and {j}: {str(e)}")
                    continue
                    
        return comparisons

def print_similarity_results(similarities):
    """Print detailed similarity results"""
    for (doc1, doc2), scores in similarities.items():
        print(f"\nSimilarity Analysis between Document {doc1} and Document {doc2}:")
        print("-" * 60)
        
        for section_idx, section_scores in enumerate(scores):
            print(f"\nSection {section_idx} comparisons:")
            for score_info in section_scores:
                print(f"  Section {score_info['section1_idx']} (Doc {doc1}) vs "
                      f"Section {score_info['section2_idx']} (Doc {doc2}): "
                      f"{score_info['similarity']:.3f}")
            
            # Print average similarity for this section
            avg_similarity = sum(s['similarity'] for s in section_scores) / len(section_scores)
            print(f"\n  Average similarity for Section {section_idx}: {avg_similarity:.3f}")
        
        # Print overall document similarity
        all_scores = [s['similarity'] for section in scores for s in section]
        overall_similarity = sum(all_scores) / len(all_scores)
        print(f"\nOverall Document Similarity: {overall_similarity:.3f}")
        print("=" * 60)

# Usage:
# Usage
document_paths = [
    "data/1.jpeg",
    "data/2.jpeg",
    "data/1.jpeg"
]

processor = DocumentProcessor()
print("Starting document processing...")
similarities = processor.process_and_compare_documents(document_paths)
print("\nSimilarity Results:")
print_similarity_results(similarities)