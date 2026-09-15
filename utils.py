
from ScanDL2.scandl_module.original_scandl.utils.dist_util import load_state_dict
from ScanDL2.fix_dur_module.utils_data import padding_and_mask_seq2seq, aggregate_input_embeddings 
from ScanDL2.scandl2_utils import TextDataset
from ScanDL2.scandl2_utils import text_dataset_loader
from ScanDL2.scandl2_utils import FixdurDataset
from ScanDL2.scandl_module.scripts.sp_basic_utils import  create_model_and_diffusion, load_defaults_config, add_dict_to_argparser,args_to_dict

from ScanDL2.fix_dur_module.utils_data import get_embeddings_seq2seq,prepare_seq2seq_data,get_embeddings_seq2seq_hp,prepare_seq2seq_data_hp, Seq2SeqDataset,Seq2SeqDatasetHP 

from ScanDL2.fix_dur_module.scasim import scasim

from ScanDL2.model import ScanDL2
from ScanDL2.model import ScanDLModule

__all__=[

    'padding_and_mask_seq2seq',
    'aggregate_input_embeddings',
    'TextDataset',
    'text_dataset_loader',
    'FixdurDataset',
    'load_defaults_config',
    'add_dict_to_argparser',
    'args_to_dict',
    'get_embeddings_seq2seq',
    'prepare_seq2seq_data',
    'Seq2SeqDataset',
    'scasim',  
    'create_model_and_diffusion', 
    'load_state_dict',             
    
]



def get_scanpath(input,model_params,input_type="text",input_format="sentence",model="ScanDL2"):

    res = {}

    if model == "ScanDL2":
        model = ScanDL2() 
    elif model == "ScanDL":
        model = ScanDLModule()

    if input_type == "text":

        out = model(
            text_type='sentence',
            bsz=2,
            save='scandl2_pkg/test_sentences',
            filename='sample_output',
        )

    else:
        pass
    


    return