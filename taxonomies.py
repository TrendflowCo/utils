import pickle as pkl
from pathlib import Path

curr_path = Path(__file__).parent.resolve()

# CLIP
with open(curr_path / Path('vocab/taxonomy_embeddings_clip.pkl'), 'rb') as f:
    taxonomy_clip = pkl.load(f)

with open(curr_path / Path('vocab/attributes_embeddings_clip.pkl'), 'rb') as f:
    attributes_clip = pkl.load(f)
 
with open(curr_path / Path('vocab/db/taxonomy_embeddings_clip.pkl'), 'rb') as f:
    taxonomy_db_clip = pkl.load(f)


with open(curr_path / Path('vocab/db/garment_tags_fclip_text.pkl'), 'rb') as f:
    garment_tags_fclip_text = pkl.load(f)
    
with open(curr_path / Path('vocab/db/cat_attrs_fclip_dict.pkl'), 'rb') as f:
    cat_attrs_fclip_dict = pkl.load(f)
  
# FCLIP
with open(curr_path / Path('vocab/taxonomy_embeddings_fclip.pkl'), 'rb') as f:
    taxonomy_fclip = pkl.load(f)

with open(curr_path / Path('vocab/attributes_embeddings_fclip.pkl'), 'rb') as f:
    attributes_fclip = pkl.load(f)


with open(curr_path / Path('vocab/attributes_embeddings_clip_flat.pkl'), 'rb') as f:
    attributes_clip_text = pkl.load(f)

with open(curr_path / Path('vocab/attributes_embeddings_fclip_flat.pkl'), 'rb') as f:
    attributes_fclip_text = pkl.load(f)
    
with open(curr_path / Path('vocab/db/taxonomy_embeddings_fclip.pkl'), 'rb') as f:
    taxonomy_db_fclip = pkl.load(f)
    