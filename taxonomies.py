import pickle as pkl

# CLIP
with open('./taxonomies/taxonomy_embeddings_clip.pkl', 'rb') as f:
    taxonomy_clip = pkl.load(f)

with open('./taxonomies/attributes_embeddings_clip.pkl', 'rb') as f:
    attributes_clip = pkl.load(f)


# FCLIP
with open('./taxonomies/taxonomy_embeddings_fclip.pkl', 'rb') as f:
    taxonomy_fclip = pkl.load(f)

with open('./taxonomies/attributes_embeddings_fclip.pkl', 'rb') as f:
    attributes_fclip = pkl.load(f)


with open('./taxonomies/attributes_embeddings_clip_flat.pkl', 'rb') as f:
    attributes_clip_text = pkl.load(f)

with open('./taxonomies/attributes_embeddings_fclip_flat.pkl', 'rb') as f:
    attributes_fclip_text = pkl.load(f)
    
with open('./taxonomies/db/taxonomy_embeddings_clip.pkl', 'rb') as f:
    taxonomy_db_clip = pkl.load(f)