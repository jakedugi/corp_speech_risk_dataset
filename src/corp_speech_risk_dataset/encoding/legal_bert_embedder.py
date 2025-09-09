"""
Legal-BERT embeddings module for corporate speech risk analysis.

This module provides Legal-BERT embeddings optimized for T4 GPU with:
- Batch processing for efficient VRAM usage
- Automatic Mixed Precision (AMP) for 30% speedup
- Mean pooling for sentence-level representations
- Device-agnostic operation (CUDA → MPS → CPU priority)
"""

from typing import List, Optional
import torch
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, AutoModel
import time


class LegalBertEmbedder:
    """
    Legal-BERT embedder optimized for T4 GPU efficiency.

    Features:
    - Batch processing with configurable batch size (default: 32 for T4's 16GB VRAM)
    - Automatic Mixed Precision (AMP) for ~30% memory reduction and speedup
    - Mean pooling over attention-masked tokens for sentence representations
    - Handles long legal texts via truncation at 512 tokens
    - 768-dimensional output embeddings
    """

    def __init__(
        self,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ):
        """
        Initialize Legal-BERT embedder.

        Args:
            model_name: HuggingFace model name for Legal-BERT
            device: Target device (auto-detected if None)
            use_amp: Enable Automatic Mixed Precision for efficiency
        """
        self.model_name = model_name
        self.use_amp = use_amp

        # Device selection with consistent priority: CUDA → MPS → CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        logger.info(f"Legal-BERT initializing on device: {self.device}")

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.success(f"✓ Legal-BERT loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load Legal-BERT model {self.model_name}: {e}")
            raise

    def get_sentence_embedding_dimension(self) -> int:
        """Return embedding dimension (768 for Legal-BERT base)."""
        return self.model.config.hidden_size

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate Legal-BERT embeddings with batching and AMP optimization.

        Args:
            texts: List of input texts to embed
            batch_size: Processing batch size (32 optimized for T4 GPU)
            max_length: Maximum token length (512 for efficiency)
            convert_to_numpy: Return numpy array if True, torch tensor if False
            show_progress: Print progress updates for large batches

        Returns:
            Embeddings array of shape (len(texts), 768)
        """
        if not texts:
            return np.array([]).reshape(0, self.get_sentence_embedding_dimension())

        start_time = time.time()
        embeddings = []

        logger.info(
            f"Legal-BERT encoding {len(texts)} texts with batch_size={batch_size}, "
            f"AMP={'ON' if self.use_amp else 'OFF'}"
        )

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Progress logging for large datasets
                if show_progress and i > 0 and i % (batch_size * 50) == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(texts) - i) / rate if rate > 0 else 0
                    logger.info(
                        f"Legal-BERT progress: {i:,}/{len(texts):,} "
                        f"({100*i/len(texts):.1f}%) | "
                        f"Rate: {rate:.1f} texts/s | ETA: {eta:.1f}s"
                    )

                # Tokenize batch
                try:
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    ).to(self.device)
                except Exception as e:
                    logger.warning(
                        f"Tokenization failed for batch {i//batch_size}: {e}"
                    )
                    # Fallback: create zero embeddings for this batch
                    batch_embeddings = torch.zeros(
                        len(batch_texts), self.get_sentence_embedding_dimension()
                    )
                    embeddings.append(batch_embeddings.cpu().numpy())
                    continue

                # Forward pass with optional AMP
                try:
                    if self.use_amp and self.device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)

                    # Mean pooling with attention mask
                    last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
                    attention_mask = inputs["attention_mask"]

                    # Expand mask to match hidden state dimensions
                    mask_expanded = (
                        attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    )

                    # Sum embeddings and mask, then average
                    sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    mean_pooled = sum_embeddings / sum_mask  # (batch, 768)

                    embeddings.append(mean_pooled.cpu().numpy())

                except Exception as e:
                    logger.warning(
                        f"Forward pass failed for batch {i//batch_size}: {e}"
                    )
                    # Fallback: create zero embeddings for this batch
                    batch_embeddings = torch.zeros(
                        len(batch_texts), self.get_sentence_embedding_dimension()
                    )
                    embeddings.append(batch_embeddings.cpu().numpy())

        # Combine all batch embeddings
        result = (
            np.vstack(embeddings)
            if embeddings
            else np.array([]).reshape(0, self.get_sentence_embedding_dimension())
        )

        total_time = time.time() - start_time
        rate = len(texts) / total_time if total_time > 0 else 0
        logger.success(
            f"✓ Legal-BERT encoded {len(texts):,} texts in {total_time:.2f}s "
            f"({rate:.1f} texts/s) → {result.shape}"
        )

        return result if convert_to_numpy else torch.tensor(result)


def get_legal_bert_embedder(
    model_name: str = "nlpaueb/legal-bert-base-uncased",
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> LegalBertEmbedder:
    """
    Factory function to create Legal-BERT embedder instance.

    Args:
        model_name: HuggingFace model name for Legal-BERT
        device: Target device (auto-detected if None)
        use_amp: Enable Automatic Mixed Precision

    Returns:
        Configured LegalBertEmbedder instance
    """
    return LegalBertEmbedder(model_name=model_name, device=device, use_amp=use_amp)


def get_legal_bert_embeddings(
    texts: List[str],
    batch_size: int = 32,
    use_amp: bool = True,
    model_name: str = "nlpaueb/legal-bert-base-uncased",
) -> np.ndarray:
    """
    Convenience function for one-shot Legal-BERT embedding generation.

    Args:
        texts: List of input texts
        batch_size: Processing batch size (32 for T4 efficiency)
        use_amp: Enable Automatic Mixed Precision
        model_name: HuggingFace model name

    Returns:
        Embeddings array of shape (len(texts), 768)
    """
    embedder = get_legal_bert_embedder(model_name=model_name, use_amp=use_amp)
    return embedder.encode(
        texts, batch_size=batch_size, show_progress=len(texts) > 1000
    )
