import torch
import torch.nn as nn
from peft import (
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)
from transformers import BertForSequenceClassification, DynamicCache
from transformers.modeling_outputs import SequenceClassifierOutput

from .config import MODEL_NAME


class SoftPromptWrapper(nn.Module):
    def __init__(self, model, num_tokens=20, embedding_dim=128):
        super().__init__()
        self.base_model = model
        self.num_tokens = num_tokens
        self.soft_prompt = nn.Parameter(torch.randn(num_tokens, embedding_dim))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        if hasattr(self.base_model, "bert"):
            word_embeddings = self.base_model.bert.embeddings.word_embeddings
        elif hasattr(self.base_model, "base_model"):
            word_embeddings = self.base_model.base_model.model.bert.embeddings.word_embeddings
        else:
            word_embeddings = self.base_model.get_input_embeddings()

        inputs_embeds = word_embeddings(input_ids)
        batch_size = inputs_embeds.shape[0]
        prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat((prompt_batch, inputs_embeds), dim=1)

        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prompt_mask, attention_mask), dim=1)

        if token_type_ids is not None:
            prompt_token_types = torch.zeros(batch_size, self.num_tokens, dtype=torch.long).to(token_type_ids.device)
            token_type_ids = torch.cat((prompt_token_types, token_type_ids), dim=1)

        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs,
        )


class PrefixWrapper(nn.Module):
    """Injects learned prefix past_key_values into each BERT attention layer."""

    def __init__(self, model, num_tokens=20):
        super().__init__()
        self.base_model = model
        self.num_tokens = num_tokens

        if hasattr(model, "bert"):
            bert_config = model.bert.config
        elif hasattr(model, "base_model"):
            bert_config = model.base_model.model.bert.config

        self.num_layers = bert_config.num_hidden_layers
        self.num_heads = bert_config.num_attention_heads
        self.head_dim = bert_config.hidden_size // bert_config.num_attention_heads
        hidden_size = bert_config.hidden_size

        # (num_tokens, num_layers * 2 * hidden_size) — 2 for key and value
        self.prefix_embedding = nn.Embedding(num_tokens, self.num_layers * 2 * hidden_size)

    def _get_past_key_values(self, batch_size, device):
        prefix_ids = torch.arange(self.num_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        prefix_raw = self.prefix_embedding(prefix_ids)  # (batch, num_tokens, num_layers*2*hidden)
        prefix_raw = prefix_raw.view(batch_size, self.num_tokens, self.num_layers, 2, self.num_heads, self.head_dim)
        prefix_raw = prefix_raw.permute(2, 3, 0, 4, 1, 5).contiguous()  # (num_layers, 2, batch, heads, tokens, head_dim)
        return DynamicCache([(prefix_raw[i, 0], prefix_raw[i, 1]) for i in range(self.num_layers)])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        past_key_values = self._get_past_key_values(batch_size, device)

        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, self.num_tokens, device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if hasattr(self.base_model, "bert"):
            bert = self.base_model.bert
            dropout = self.base_model.dropout
            classifier = self.base_model.classifier
        elif hasattr(self.base_model, "base_model"):
            bert = self.base_model.base_model.model.bert
            dropout = self.base_model.base_model.model.dropout
            classifier = self.base_model.base_model.model.classifier

        bert_outputs = bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
        )
        pooled_output = bert_outputs.pooler_output
        logits = classifier(dropout(pooled_output))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def get_trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def setup_model(method, num_labels):
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    if method == "full-finetuning":
        pass
    elif method == "last-layer-finetuning":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif method == "lora":
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
    elif method == "soft-prompt":
        peft_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=20)
        model = get_peft_model(model, peft_config)
    elif method == "prefix":
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=20)
        model = get_peft_model(model, peft_config)
    elif method == "soft-prompt+lora":
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model = SoftPromptWrapper(model, num_tokens=20, embedding_dim=128)
    elif method == "prefix+lora":
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, lora_config)
        model = PrefixWrapper(model, num_tokens=20)
    elif method == "ia3":
        peft_config = IA3Config(task_type=TaskType.SEQ_CLS, inference_mode=False)
        model = get_peft_model(model, peft_config)
    else:
        raise ValueError(f"Unknown method: {method}")

    return model
