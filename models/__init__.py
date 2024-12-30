from models.base import BaseModel
from models.sequential_knowledge_transformer import SKT_KG
from models.skt_kg2 import SKT_FULLKG
from models.skt import SequentialKnowledgeTransformer
from models.pipm import PIPM
from models.pipm_seq import PIPM_seq

from models.kdbts_teacher import SKT_fixteacher
from models.kdbts_student import SKT_student

from models.combination_teacher import PIPM_teacher
from models.combination_student import PIPM_student

from models.GCN import GCN
from models.attention import MultiHeadAttention

MODELS = {
    'SKT_KG': SKT_KG,
    'SKT_FULLKG': SKT_FULLKG,

    'SequentialKnowledgeTransformer': SequentialKnowledgeTransformer,
    "SKT_fixteacher":SKT_fixteacher,
    "SKT_student":SKT_student,

    "PIPM":PIPM,
    "PIPM_teacher":PIPM_teacher,
    "PIPM_student":PIPM_student,

    "PIPM_seq":PIPM_seq,
    "TODO":object,
}
