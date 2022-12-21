
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


'''
BertModel's output

if not return_dict:
    return (sequence_output, pooled_output) + encoder_outputs[1:]

return BaseModelOutputWithPoolingAndCrossAttentions(
    last_hidden_state=sequence_output,
    pooler_output=pooled_output,
    past_key_values=encoder_outputs.past_key_values,
    hidden_states=encoder_outputs.hidden_states,
    attentions=encoder_outputs.attentions,
    cross_attentions=encoder_outputs.cross_attentions,
)
'''


class BertForMultiMaskClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.cls_token_id = 101

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0][:, 1:, :] # last hidden states. doesn't count the first [CLS]
        sequence_output = self.dropout(sequence_output)
        sequence_output = torch.reshape(sequence_output, (-1, sequence_output.shape[-1])) # -> [B*L, H]
        masked_positions = torch.nonzero(input_ids[:, 1:].flatten() == self.cls_token_id)
        # print(masked_positions)
        sentence_representations = sequence_output.index_select(0, masked_positions.flatten())

        logits = self.classifier(sentence_representations)
        # print("logits", logits.shape)
        bsz, label_len = labels.shape
        # print(bsz, label_len)
        labels_flat = labels.flatten() # non mask label position
        label_indices = torch.nonzero(labels_flat!=-1).flatten()
        logit_template = labels_flat.unsqueeze(-1).expand(labels_flat.shape[0], self.num_labels).float().clone()
        # print("label indices", label_indices.shape)
        # print("logit template", logit_template.shape)

        labels = labels[labels!=-1].long() # remove padding. flatten.
        # print(logits, labels)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        
    
        # reshape logits 
        # predictions = np.argmax(logits, axis=1) 
        # print("label indices", label_indices)
        # print("logit_template", logit_template)
        logits = logit_template.index_copy_(0, label_indices, logits)
        # print("filled logits", logits)
        logits = logits.reshape(bsz, label_len, -1)
        # print("reshape logits", logits)
        
        # print("logits", logits.shape)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


