import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import math
import action_slot
from transformers import (
    AutoConfig,
    AutoModel,
)


class GRUWaypointsPredictorInterFuser(nn.Module):
	"""
	A version of the waypoint GRU used in InterFuser.
	It embeds the target point and inputs it as hidden dimension instead of input.
	The scene state is described by waypoints x input_dim features which are added as input instead of initializing the
	hidden state.
	"""

	def __init__(self, input_dim, waypoints, hidden_size, target_point_size):
		super().__init__()
		self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
		if target_point_size > 0:
			self.encoder = nn.Linear(target_point_size, hidden_size)
		self.target_point_size = target_point_size
		self.hidden_size = hidden_size
		self.decoder = nn.Linear(hidden_size, 2)
		self.waypoints = waypoints

	def forward(self, x, target_point):
		bs = x.shape[0]
		if self.target_point_size > 0:
			z = self.encoder(target_point).unsqueeze(0)
		else:
			z = torch.zeros((1, bs, self.hidden_size), device=x.device)
		output, _ = self.gru(x, z)
		output = output.reshape(bs * self.waypoints, -1)
		output = self.decoder(output).reshape(bs, self.waypoints, 2)
		output = torch.cumsum(output, 1)
		return output



class GRUWaypointsPredictorTransFuser(nn.Module):
	"""
	The waypoint GRU used in TransFuser.
	It enters the target point as input.
	The hidden state is initialized with the scene features.
	The input is autoregressive and starts either at 0 or learned.
	"""

	def __init__(self, pred_len, hidden_size, target_point_size):
		super().__init__()
		self.wp_decoder = nn.GRUCell(input_size=2 + target_point_size, hidden_size=hidden_size)
		self.output = nn.Sequential(
    	nn.Linear(hidden_size, ),
    	nn.Linear(hidden_size, 2)
		)
		self.prediction_len = pred_len
		self.target_point_size = target_point_size

	def forward(self, z, target_point):
		output_wp = []

		x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

		target_point = target_point.clone()
		# autoregressive generation of output waypoints
		for _ in range(self.prediction_len):
			if self.target_point_size > 0:
				x_in = torch.cat([x, target_point], dim=1)
			else:
				x_in = x
    
			z = self.wp_decoder(x_in, z)
			dx = self.output(z)

			x = dx + x

			output_wp.append(x)

		pred_wp = torch.stack(output_wp, dim=1)

		return pred_wp


class PositionEmbeddingSine(nn.Module):
	"""
	Taken from InterFuser
	This is a more standard version of the position embedding, very similar to the one
	used by the Attention is all you need paper, generalized to work on images.
	"""

	def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
		super().__init__()
		self.num_pos_feats = num_pos_feats
		self.temperature = temperature
		self.normalize = normalize
		if scale is not None and normalize is False:
			raise ValueError('normalize should be True if scale is passed')
		if scale is None:
			scale = 2 * math.pi
		self.scale = scale

	def forward(self, tensor):
		x = tensor
		bs, _, h, w = x.shape
		not_mask = torch.ones((bs, h, w), device=x.device)
		y_embed = not_mask.cumsum(1, dtype=torch.float32)
		x_embed = not_mask.cumsum(2, dtype=torch.float32)
		if self.normalize:
			eps = 1e-6
			y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
			x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

		dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
		dim_t = self.temperature**(2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

		pos_x = x_embed[:, :, :, None] / dim_t
		pos_y = y_embed[:, :, :, None] / dim_t
		pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
		pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
		pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
		return pos


class MultiheadAttentionWithAttention(nn.Module):
	"""
	MultiheadAttention that also return attention weights
	"""
	def __init__(self, n_embd, n_head, pdrop):
		super().__init__()
		assert n_embd % n_head == 0
		# key, query, value projections for all heads
		self.key = nn.Linear(n_embd, n_embd)
		self.query = nn.Linear(n_embd, n_embd)
		self.value = nn.Linear(n_embd, n_embd)
		# regularization
		self.attn_drop = nn.Dropout(pdrop)
		self.resid_drop = nn.Dropout(pdrop)
		# output projection
		self.proj = nn.Linear(n_embd, n_embd)
		self.n_head = n_head

	def forward(self, q_in, k_in, v_in):
		b, t, c = q_in.size()
		_, t_mem, _ = k_in.size()

		# calculate query, key, values for all heads in batch and move head
		# forward to be the batch dim
		q = self.query(q_in).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
		k = self.key(k_in).view(b, t_mem, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
		v = self.value(v_in).view(b, t_mem, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

		# self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
		att = F.softmax(att, dim=-1)
		att = self.attn_drop(att)
		y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
		y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

		# output projection
		y = self.resid_drop(self.proj(y))
		attention = torch.mean(att, dim=1)  # Average attention over heads
		return y, attention


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_positions, embedding_dim):
        """
        Learnable positional embeddings.
        
        :param num_positions: Number of positions (sequence length)
        :param embedding_dim: Dimensionality of each positional embedding
        """
        super().__init__()
        # Learnable positional embeddings for each position
        self.positional_embeddings = nn.Embedding(num_positions, embedding_dim)

    def forward(self, input_tensor):
        """
        Adds positional embeddings to the input tensor.
        
        :param input_tensor: Input tensor of shape (batch_size, num_positions, embedding_dim)
        :return: Tensor with added positional embeddings
        """
        # Get positions (0 to num_positions-1) for the sequence
        batch_size, num_positions, _ = input_tensor.size()
        positions = torch.arange(num_positions, device=input_tensor.device).unsqueeze(0)
        positions = positions.expand(batch_size, num_positions)  # Shape: (batch_size, num_positions)

        # Add positional embeddings
        return input_tensor + self.positional_embeddings(positions)


class ActionSlotPlanner(nn.Module):
	def __init__(self, args, num_ego_class, num_actor_class, num_slots=21, box=False, videomae=None):
		super().__init__()
		self.args = args
		self.action_slot = action_slot.ACTION_SLOT(args, num_ego_class, num_actor_class, args.num_slots, box=args.box)
		self.init = False
  
		if args.use_tp:
			targert_point_size = 2
		else:
			targert_point_size = 0

		extra_size = 0
		if args.use_velocity:
			extra_sensor_channels = args.gru_input_size
			self.velocity_normalization = nn.BatchNorm1d(1, affine=False)
			extra_size += 1
			self.extra_sensor_encoder = nn.Sequential(nn.Linear(extra_size, 128), nn.ReLU(inplace=True),
													nn.Linear(128, extra_sensor_channels), nn.ReLU(inplace=True))   
		else:
			extra_sensor_channels = args.extra_sensor_channels

		self.pe = LearnablePositionalEmbedding(args.num_slots + extra_size, args.channel)
   
		if args.transformer_decoder_join:
			decoder_norm = nn.LayerNorm(args.gru_input_size)
			decoder_layer = nn.TransformerDecoderLayer(self.args.gru_input_size,
														self.args.num_decoder_heads,
														activation=nn.GELU(),
														batch_first=True)
			self.join = torch.nn.TransformerDecoder(decoder_layer,
													num_layers=self.args.num_transformer_decoder_layers,
													norm=decoder_norm)


			# We don't have an encoder, so we directly use it on the features
			self.encoder_pos_encoding = PositionEmbeddingSine(args.gru_input_size // 2, normalize=True)
			# self.extra_sensor_pos_embed = nn.Parameter(torch.zeros(1, args.gru_input_size))

			self.wp_query = nn.Parameter(
					torch.zeros(1, (args.pred_len), args.gru_input_size))

			self.wp_decoder = GRUWaypointsPredictorInterFuser(input_dim=args.gru_input_size,
															hidden_size=args.gru_hidden_size,
															waypoints=(args.pred_len),
															target_point_size=targert_point_size)

		elif args.plant:
			trans_out_features = 512
			if self.args.use_velocity:
				trans_out_features = 512 + extra_sensor_channels

			auto_config = AutoConfig.from_pretrained(args.plant_hf_checkpoint)
			n_embd = auto_config.hidden_size
			self.model = AutoModel.from_config(config=auto_config)

			self.cls_emb = nn.Parameter(torch.randn(1, args.channel))

			# wp (CLS) decoding
			self.wp_head = nn.Linear(trans_out_features, args.gru_hidden_size)

			self.wp_decoder = GRUWaypointsPredictorTransFuser(args.pred_len, args.gru_hidden_size, targert_point_size)

			self.wp_decoder = nn.GRUCell(input_size=2 + targert_point_size, hidden_size=64)
			self.wp_output = nn.Linear(64, 2)


		else:
			join_output_features = args.gru_hidden_size
			# waypoints prediction
			self.join = nn.Sequential(
					nn.Linear(args.channel, 128),
					nn.ReLU(inplace=True),
					nn.Linear(128, join_output_features),
					nn.ReLU(inplace=True),
			)
			self.wp_decoder = GRUWaypointsPredictorTransFuser(args.pred_len, args.gru_hidden_size * (args.num_slots + extra_size), targert_point_size)


	def _init(self, x):
		self.action_slot._init(x)
		self.init = True



	def forward(self, x, ego_vel, target_point, slot_feature=None):
		if not self.init:
			self._init(x)
  
		if self.args.gen_feat:
			I3D_feat, slot_feat = self.action_slot(x)
			return I3D_feat, slot_feat
 
		if slot_feature is None:
			slot_feature, attn_masks = self.action_slot(x)
  
		bs = ego_vel.shape[0]
  
		if self.args.transformer_decoder_join:
			slot_feature = slot_feature.permute(0,2,1)
			if self.args.use_velocity:
				extra_sensors = self.velocity_normalization(ego_vel)
				extra_sensors = self.extra_sensor_encoder(extra_sensors)
		
				if self.args.transformer_decoder_join:
					# extra_sensors = extra_sensors + self.extra_sensor_pos_embed.repeat(bs, 1)
					slot_feature = torch.cat((slot_feature, extra_sensors.unsqueeze(2)), axis=2)
			
			slot_feature = torch.permute(slot_feature, (0, 2, 1))
			slot_feature = self.pe(slot_feature)
   
			joined_wp_features = self.join(self.wp_query.repeat(bs, 1, 1), slot_feature)
			pred_wp = self.wp_decoder(joined_wp_features, target_point)
   
		elif self.args.plant:
			cls_token = self.cls_emb.repeat(bs, 1, 1)
			x = torch.cat((cls_token, slot_feature), dim=1)
   
			x = self.pe(x)
	    # Transformer Encoder; use embedding for hugging face model and get output states and attention map
			output = self.model(**{'inputs_embeds': x}, output_attentions=True)
			tf_features = output.last_hidden_state
			cls_feature = tf_features[:, 0, :]
			if self.config.use_velocity:
				normalized_velocity = self.velocity_normalization(ego_vel)
				velocity_embedding = self.extra_sensor_encoder(normalized_velocity)
				cls_feature = torch.cat((cls_feature, velocity_embedding), axis=1)

			z = self.wp_head(cls_feature)
			pred_wp = self.wp_decoder(z, target_point)


		else:
			joined_features = self.join(slot_feature)
			print(joined_features.shape)
			gru_features = torch.flatten(joined_features, 1)
			print(gru_features.shape)
			pred_wp = self.wp_decoder(gru_features, target_point)

		return pred_wp