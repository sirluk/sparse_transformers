### 2nd way to add diff parametrizations

#     @torch.no_grad()
#     def _add_diff_parametrizations(self, n_parametrizations: int = 1, p_requires_grad: bool = False, fixmask_init: bool = False, **kwargs) -> None:
#         assert not self._parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
        
#         if fixmask_init:
#             for base_module in self.get_encoder_base_modules():
#                 for n,p in list(base_module.named_parameters()):
#                     p.requires_grad = p_requires_grad
#                     p.requires_grad = p_requires_grad
#                     for _ in range(n_parametrizations):
#                         parametrize.register_parametrization(base_module, n, DiffWeightFixmask(
#                             torch.zeros_like(p), torch.ones_like(p, dtype=bool)
#                         ))
#             self._model_state = ModelState.FIXMASK
#         else:
#             for base_module in self.get_encoder_base_modules():
#                 module_copy = copy.deepcopy(base_module)
#                 if not hasattr(module_copy, "reset_parameters"):
#                     raise Exception(f"Module of type {type(module_copy)} has no attribute 'reset_parameters'")
                
#                 named_params = list(base_module.named_parameters())
#                 for _ in range(n_parametrizations): # number of diff networks to add
#                     module_copy.reset_parameters()
#                     for n,p in named_params:
#                         p.requires_grad = p_requires_grad
#                         p_init = getattr(module_copy, n)
#                         parametrize.register_parametrization(base_module, n, DiffWeightFinetune(p_init, **kwargs))
                
#             self._model_state = ModelState.FINETUNING


### norec dataloader


# def get_data_loader_norec(tokenizer, data_path, labels_task_path, labels_prot_path=None, 
#                      batch_size=16, max_length=512, raw=False, shuffle=True, debug=False):
    
#     with open(data_path, 'rb') as file:
#         data = pickle.load(file)
        
#     if debug:
#         cutoff = min(int(batch_size*10), len(data))
#         data = data[:cutoff]

#     keys = ["idx", "text", "rating", "gender"]
#     data_dict = {k:[d.get(k) for d in data] for k in keys}
    
#     input_ids, token_type_ids, attention_masks = multiprocess_tokenization(list(data_dict["text"]), tokenizer, max_length)
    
#     labels_task = read_label_file(labels_task_path)
#     labels_task = torch.tensor([labels_task[str(t)] for t in data_dict["rating"]], dtype=torch.long)
    
#     tds = [
#         input_ids,
#         token_type_ids,
#         attention_masks,
#         labels_task
#     ]
    
#     if labels_prot_path:
#         labels_prot = read_label_file(labels_prot_path)
#         tds.append(torch.tensor([labels_prot[t] for t in data_dict["gender"]], dtype=torch.long))
#         collate_fn = batch_fn_prot
#     else:
#         collate_fn = batch_fn
    
#     _dataset = TensorDataset(*tds)

#     _loader = DataLoader(_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=False, collate_fn=collate_fn)

#     return _loader



### ### ###
# class DiffWeightFinetune(nn.Module):

#     def __init__(self, weight, alpha_init, concrete_lower, concrete_upper, structured):
#         super().__init__()
#         self.concrete_lower = concrete_lower
#         self.concrete_upper = concrete_upper
#         self.structured = structured

#         self.register_parameter("diff_weight", Parameter(weight))
#         self.register_parameter("alpha", Parameter(torch.zeros_like(weight) + alpha_init))

#         if structured:
#             self.register_parameter("alpha_group", Parameter(torch.zeros((1,), device=weight.device) + alpha_init))

#         self.active = True

#     def forward(self, X):
#         if self.active:
#             return X + self.z * self.diff_weight
#         else:
#             return X

#     @property
#     def z(self) -> Parameter:
#         z = self.dist(self.alpha)
#         if self.structured:
#             z *= self.dist(self.alpha_group)
#         return z

#     @property
#     def alpha_weights(self) -> list:
#         alpha = [self.alpha]
#         if self.structured:
#             alpha.append(self.alpha_group)
#         return alpha

#     def dist(self, alpha) -> torch.Tensor:
#         return concrete_stretched(
#             alpha,
#             l=self.concrete_lower,
#             r=self.concrete_upper,
#             deterministic=(not self.training)
#         )

#     def set_frozen(self, frozen: bool) -> None:
#         self.diff_weight.requires_grad = not frozen
#         self.alpha.requires_grad = not frozen
#         if self.structured:
#             self.alpha_group.requires_grad = not frozen
#         if frozen:
#             self.eval()
#         else:
#             self.train()