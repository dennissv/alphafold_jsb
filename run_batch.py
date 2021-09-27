#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:24:38 2021

@author: dennis
"""
#@title Input protein sequence, then hit `Runtime` -> `Run all`
import os
import shutil
import os.path
import re
import hashlib
import time

def add_hash(x,y):
  return x

input_dir = 'ready' #@param {type:"string"}
done_dir = 'done'
result_dir = 'results' #@param {type:"string"}

# create directories
os.makedirs(result_dir, exist_ok=True)
os.makedirs(done_dir, exist_ok=True)

# number of models to use
#@markdown ---
#@markdown ### Advanced settings
msa_mode = "MMseqs2 (UniRef+Environmental)" #@param ["MMseqs2 (UniRef+Environmental)", "MMseqs2 (UniRef only)","single_sequence","custom"]
num_models = 1 #@param [1,2,3,4,5] {type:"raw"}
use_msa = True if msa_mode.startswith("MMseqs2") else False
use_env = True if msa_mode == "MMseqs2 (UniRef+Environmental)" else False
use_custom_msa = False
use_amber = False #@param {type:"boolean"}
use_templates = False #@param {type:"boolean"}
do_not_overwite_results = True #@param {type:"boolean"}

homooligomer = 1 
with open(f"run.log", "w") as text_file:
    text_file.write("num_models=%s\n" % num_models)
    text_file.write("use_amber=%s\n" % use_amber)
    text_file.write("use_msa=%s\n" % use_msa)
    text_file.write("msa_mode=%s\n" % msa_mode)
    text_file.write("use_templates=%s\n" % use_templates)



# hiding warning messages
import warnings
from absl import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

warnings.filterwarnings('ignore')
logging.set_verbosity("error")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import sys
import numpy as np
import pickle
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data.tools import hhsearch
from alphafold.model.tf import shape_placeholders
NUM_RES = shape_placeholders.NUM_RES
NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ
NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES

import colabfold as cf
# plotting libraries
# import py3Dmol
import matplotlib.pyplot as plt
# import ipywidgets
# from ipywidgets import interact, fixed, GridspecLayout, Output


if use_amber and "relax" not in dir():
  sys.path.insert(0, '/usr/local/lib/python3.7/site-packages/')
  from alphafold.relax import relax

def mk_mock_template(query_sequence, num_temp=1):
  ln = len(query_sequence)
  output_templates_sequence = "A"*ln
  output_confidence_scores = np.full(ln,1.0)
  templates_all_atom_positions = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
  templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
  templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                    templates.residue_constants.HHBLITS_AA_TO_ID)
  template_features = {'template_all_atom_positions': np.tile(templates_all_atom_positions[None],[num_temp,1,1,1]),
                       'template_all_atom_masks': np.tile(templates_all_atom_masks[None],[num_temp,1,1]),
                       'template_sequence': [f'none'.encode()]*num_temp,
                       'template_aatype': np.tile(np.array(templates_aatype)[None],[num_temp,1,1]),
                       'template_confidence_scores': np.tile(output_confidence_scores[None],[num_temp,1]),
                       'template_domain_names': [f'none'.encode()]*num_temp,
                       'template_release_date': [f'none'.encode()]*num_temp}
  return template_features


def mk_template(a3m_lines, template_paths):
  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=template_paths,
      max_template_date="2100-01-01",
      max_hits=20,
      kalign_binary_path="kalign",
      release_dates_path=None,
      obsolete_pdbs_path=None)

  hhsearch_pdb70_runner = hhsearch.HHSearch(binary_path="hhsearch", databases=[f"{template_paths}/pdb70"])

  hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
  hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
  templates_result = template_featurizer.get_templates(query_sequence=query_sequence,
                                                       query_pdb_code=None,
                                                       query_release_date=None,
                                                       hits=hhsearch_hits)
  return templates_result.features

def make_fixed_size(protein, shape_schema, msa_cluster_size, extra_msa_size,
                   num_res, num_templates=0):
  """Guess at the MSA and sequence dimensions to make fixed size."""

  pad_size_map = {
      NUM_RES: num_res,
      NUM_MSA_SEQ: msa_cluster_size,
      NUM_EXTRA_SEQ: extra_msa_size,
      NUM_TEMPLATES: num_templates,
  }

  for k, v in protein.items():
    # Don't transfer this to the accelerator.
    if k == 'extra_cluster_assignment':
      continue
    shape = list(v.shape)
    
    schema = shape_schema[k]

    assert len(shape) == len(schema), (
        f'Rank mismatch between shape and shape schema for {k}: '
        f'{shape} vs {schema}')
    pad_size = [
        pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
    ]
    padding = [(0, p - tf.shape(v)[i]) for i, p in enumerate(pad_size)]

    if padding:
      protein[k] = tf.pad(
          v, padding, name=f'pad_to_fixed_{k}')
      protein[k].set_shape(pad_size)
  return {k:np.asarray(v) for k,v in protein.items()}

def set_bfactor(pdb_filename, bfac, idx_res, chains):
  I = open(pdb_filename,"r").readlines()
  O = open(pdb_filename,"w")
  for line in I:
    if line[0:6] == "ATOM  ":
      seq_id = int(line[22:26].strip()) - 1
      seq_id = np.where(idx_res == seq_id)[0][0]
      O.write(f"{line[:21]}{chains[seq_id]}{line[22:60]}{bfac[seq_id]:6.2f}{line[66:]}")
  O.close()

def predict_structure(prefix, feature_dict, Ls, crop_len, model_params, use_model, do_relax=False, random_seed=0):  
  """Predicts structure using AlphaFold for the given sequence."""
  idx_res = feature_dict['residue_index']
  chains = list("".join([ascii_uppercase[n]*L for n,L in enumerate(Ls)]))

  # Run the models.
  plddts,paes = [],[]
  unrelaxed_pdb_lines = []
  relaxed_pdb_lines = []
  seq_len = feature_dict['seq_length'][0]
  for model_name, params in model_params.items():
    if model_name in use_model:
      print(f"running {model_name}")
      # swap params to avoid recompiling
      # note: models 1,2 have diff number of params compared to models 3,4,5
      if any(str(m) in model_name for m in [1,2]): model_runner = model_runner_1
      if any(str(m) in model_name for m in [3,4,5]): model_runner = model_runner_3
      model_runner.params = params

      processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
      model_config = model_runner.config
      eval_cfg = model_config.data.eval
      crop_feats = {k:[None]+v for k,v in dict(eval_cfg.feat).items()}     

      # templates models
      if model_name == "model_1" or model_name == "model_2":
        pad_msa_clusters = eval_cfg.max_msa_clusters - eval_cfg.max_templates
      else:
        pad_msa_clusters = eval_cfg.max_msa_clusters

      max_msa_clusters = pad_msa_clusters

      # let's try pad (num_res + X) 
      input_fix = make_fixed_size(processed_feature_dict,
                                  crop_feats,
                                  msa_cluster_size=max_msa_clusters, # true_msa (4, 512, 68)
                                  extra_msa_size=5120, # extra_msa (4, 5120, 68)
                                  num_res=crop_len, # aatype (4, 68)
                                  num_templates=4) # template_mask (4, 4) second value

      prediction_result = model_runner.predict(input_fix)
      unrelaxed_protein = protein.from_prediction(input_fix,prediction_result)
      unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
      plddts.append(prediction_result['plddt'][:seq_len])
      paes_res = []
      for i in range(seq_len):
        paes_res.append(prediction_result['predicted_aligned_error'][i][:seq_len])
      paes.append(paes_res)
      if do_relax:
        # Relax the prediction.
        amber_relaxer = relax.AmberRelaxation(max_iterations=0,tolerance=2.39,
                                              stiffness=10.0,exclude_residues=[],
                                              max_outer_iterations=20)      
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
        relaxed_pdb_lines.append(relaxed_pdb_str)

  # rerank models based on predicted lddt
  lddt_rank = np.mean(plddts,-1).argsort()[::-1]
  out = {}
  print("reranking models based on avg. predicted lDDT")
  os.makedirs(f'results/{prefix}', exist_ok=True)
  for n,r in enumerate(lddt_rank):
    print(f"model_{n+1} {np.mean(plddts[r])}")
    
    unrelaxed_pdb_path = f'results/{prefix}/{prefix}_unrelaxed_model_{n+1}.pdb'    
    with open(unrelaxed_pdb_path, 'w') as f: f.write(unrelaxed_pdb_lines[r])
    set_bfactor(unrelaxed_pdb_path, plddts[r], idx_res, chains)

    if do_relax:
      relaxed_pdb_path = f'results/{prefix}/{prefix}_relaxed_model_{n+1}.pdb'
      with open(relaxed_pdb_path, 'w') as f: f.write(relaxed_pdb_lines[r])
      set_bfactor(relaxed_pdb_path, plddts[r], idx_res, chains)

    out[f"model_{n+1}"] = {"plddt":plddts[r], "pae":paes[r]}
  return out

# from os import listdir
# from os.path import isfile, join
# onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
# queries = []
# for filename in onlyfiles:
#     extension = os.path.splitext(filename)[1]
#     jobname=os.path.splitext(filename)[0]
#     filepath = input_dir+"/"+filename
#     with open(filepath) as f:
#         input_fasta_str = f.read()
#     (seqs, header) = pipeline.parsers.parse_fasta(input_fasta_str)
#     query_sequence = seqs[0]
#     queries.append((jobname, query_sequence, extension))
# # sort by seq. len    
# queries.sort(key=lambda t: len(t[1]))
# import math
# crop_len = math.ceil(len(queries[0][1]) * 1.1)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

proteins = load_obj('enccn_proteins.pkl')

from glob import glob
import math

while 1:
    ready = glob(f'{input_dir}/*.a3m')
    if len(ready) < 10:
        time.sleep(300)
        continue
    queries = []
    for filepath in ready:
        extension = '.a3m'
        jobname = filepath[len(input_dir)+1:-4]
        with open(filepath) as f:
            input_fasta_str = f.read()
        query_sequence = proteins[jobname]
        queries.append((jobname, query_sequence, extension))
    queries.sort(key=lambda t: len(t[1]))
    queries = queries[:10]
    crop_len = len(queries[-1]) + 1
    
    for (jobname, query_sequence, extension) in queries:
      start = time.time()
      a3m_file = f"{jobname}.a3m"
      if len(query_sequence) > crop_len:
        crop_len = math.ceil(len(query_sequence) * 1.1)
      print("Running: "+jobname)
      if do_not_overwite_results == True and os.path.isfile(result_dir+"/"+jobname+".result.zip"):
        continue
      if use_templates:
        try:  
          a3m_lines, template_paths = cf.run_mmseqs2(query_sequence, jobname, use_env, use_templates=True)
        except:
          print(jobname+" cound not be processed")
          continue  
        if template_paths is None:
          template_features = mk_mock_template(query_sequence, 100)
        else:
          template_features = mk_template(a3m_lines, template_paths)
        if extension.lower() == ".a3m":
          a3m_lines = "".join(open(input_dir+"/"+a3m_file,"r").read())
      else:
        if extension.lower() == ".a3m":
          a3m_lines = "".join(open(input_dir+"/"+a3m_file,"r").read())
        else:
          try:
            a3m_lines = cf.run_mmseqs2(query_sequence, jobname, use_env)
          except:
            print(jobname+" cound not be processed")
            continue
        template_features = mk_mock_template(query_sequence, 100)
    
      # with open(a3m_file, "w") as text_file:
      #   text_file.write(a3m_lines)
      # parse MSA
      msa, deletion_matrix = pipeline.parsers.parse_a3m(a3m_lines)
    
      #@title Gather input features, predict structure
      from string import ascii_uppercase
    
      # collect model weights
      use_model = {}
      if "model_params" not in dir(): model_params = {}
      for model_name in ["model_1","model_2","model_3","model_4","model_5"][:num_models]:
        use_model[model_name] = True
        if model_name not in model_params:
          model_params[model_name] = data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir=".")
          if model_name == "model_1":
            model_config = config.model_config(model_name+"_ptm")
            model_config.data.eval.num_ensemble = 1
            model_runner_1 = model.RunModel(model_config, model_params[model_name])
          if model_name == "model_3":
            model_config = config.model_config(model_name+"_ptm")
            model_config.data.eval.num_ensemble = 1
            model_runner_3 = model.RunModel(model_config, model_params[model_name])
    
    
      msas = [msa]
      deletion_matrices = [deletion_matrix]
      try:
      # gather features
        feature_dict = {
            **pipeline.make_sequence_features(sequence=query_sequence,
                                              description="none",
                                              num_res=len(query_sequence)),
            **pipeline.make_msa_features(msas=msas,deletion_matrices=deletion_matrices),
            **template_features
        }
      except:
        print(jobname+" cound not be processed")
        continue
    
      
      outs = predict_structure(jobname, feature_dict,
                               Ls=[len(query_sequence)], crop_len=crop_len,
                               model_params=model_params, use_model=use_model,
                               do_relax=use_amber)
    
      # gather MSA info
      deduped_full_msa = list(dict.fromkeys(msa))
      msa_arr = np.array([list(seq) for seq in deduped_full_msa])
      seqid = (np.array(list(query_sequence)) == msa_arr).mean(-1)
      seqid_sort = seqid.argsort() #[::-1]
      non_gaps = (msa_arr != "-").astype(float)
      non_gaps[non_gaps == 0] = np.nan
    
      ##################################################################
      plt.figure(figsize=(14,4),dpi=100)
      ##################################################################
      plt.subplot(1,2,1); plt.title("Sequence coverage")
      plt.imshow(non_gaps[seqid_sort]*seqid[seqid_sort,None],
                 interpolation='nearest', aspect='auto',
                 cmap="rainbow_r", vmin=0, vmax=1, origin='lower')
      plt.plot((msa_arr != "-").sum(0), color='black')
      plt.xlim(-0.5,msa_arr.shape[1]-0.5)
      plt.ylim(-0.5,msa_arr.shape[0]-0.5)
      plt.colorbar(label="Sequence identity to query",)
      plt.xlabel("Positions")
      plt.ylabel("Sequences")
    
      ##################################################################
      plt.subplot(1,2,2); plt.title("Predicted lDDT per position")
      for model_name,value in outs.items():
        plt.plot(value["plddt"],label=model_name)
      if homooligomer > 0:
        for n in range(homooligomer+1):
          x = n*(len(query_sequence)-1)
          plt.plot([x,x],[0,100],color="black")
      plt.legend()
      plt.ylim(0,100)
      plt.ylabel("Predicted lDDT")
      plt.xlabel("Positions")
      plt.savefig(f'results/{jobname}/{jobname}_coverage_lDDT.png')
      ##################################################################
    
      print("Predicted Alignment Error")
      ##################################################################
      plt.figure(figsize=(3*num_models,2), dpi=100)
      for n,(model_name,value) in enumerate(outs.items()):
        plt.subplot(1,num_models,n+1)
        plt.title(model_name)
        plt.imshow(value["pae"],label=model_name,cmap="bwr",vmin=0,vmax=30)
        plt.colorbar()
      plt.savefig(f'results/{jobname}/{jobname}_PAE.png')
      ##################################################################
      
      # Move files when job is finished
      shutil.move(f'{input_dir}/{jobname}.a3m', f'{done_dir}/{jobname}.a3m')
      
      end = time.time()
      print(f'Processed {jobname} in {int(end-start)} s')
