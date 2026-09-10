"""
IMPROVED ASL - Configuration File
==================================
Defines all classes, groups, paths, and search database
Total: 65 classes (36 signs + 29 alphabets)
Organized into 13 groups (6 alphabet + 7 sign groups)
Smart grouping ensures NO similar signs in same group
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"

# Signs dataset path
SIGNS_FRAMES_DIR = DATASET_DIR / "frames_20"

# Alphabets dataset path
ALPHABETS_DIR = DATASET_DIR / "archive" / "asl_alphabet_train" / "asl_alphabet_train"

# Output directories (will be created)
KEYPOINTS_DIR = BASE_DIR / "keypoints_extracted"
MODELS_DIR = BASE_DIR / "Models"
LOGS_DIR = BASE_DIR / "logs"

# ═══════════════════════════════════════════════════════════════════════════
# CLASS DEFINITIONS (65 TOTAL CLASSES)
# ═══════════════════════════════════════════════════════════════════════════

# 36 SIGN CLASSES
SIGN_CLASSES = {
    0: 'before',
    1: 'computer',
    2: 'cook',
    3: 'cool',
    4: 'dance',
    5: 'dog',
    6: 'drink',
    7: 'eat',
    8: 'enjoy',
    9: 'family',
    10: 'fine',
    11: 'finish',
    12: 'give',
    13: 'go',
    14: 'help',
    15: 'how',
    16: 'later',
    17: 'like',
    18: 'man',
    19: 'meet',
    20: 'mother',
    21: 'need',
    22: 'no',
    23: 'now',
    24: 'play',
    25: 'school',
    26: 'son',
    27: 'tell',
    28: 'walk',
    29: 'want',
    30: 'what',
    31: 'who',
    32: 'woman',
    33: 'work',
    34: 'wrong',
    35: 'yes',
}

# 29 ALPHABET CLASSES
ALPHABET_CLASSES = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'space', 27: 'nothing', 28: 'del',
}

# ═══════════════════════════════════════════════════════════════════════════
# SMART GROUPING (13 GROUPS: 6 ALPHABET + 7 SIGN)
# ═══════════════════════════════════════════════════════════════════════════
# Strategy: NO similar signs in same group
# Similar pairs (SEPARATED):
# - help ↔ want (both requesting)
# - like ↔ enjoy (both emotions)
# - go ↔ walk (both movement)
# - yes ↔ no (opposites)
# - mother ↔ son (family)
# - man ↔ woman (gender)
# - cool ↔ enjoy (emotions)
# - finish ↔ no (negation concepts)

GROUPS = {
    # ALPHABET GROUPS (A-Z organized alphabetically)
    'ALPHA1': {
        'name': 'Alphabet Group 1',
        'type': 'alphabet',
        'classes': ['A', 'B', 'C', 'D', 'E'],
        'description': 'Letters A-E'
    },
    'ALPHA2': {
        'name': 'Alphabet Group 2',
        'type': 'alphabet',
        'classes': ['F', 'G', 'H', 'I', 'J'],
        'description': 'Letters F-J'
    },
    'ALPHA3': {
        'name': 'Alphabet Group 3',
        'type': 'alphabet',
        'classes': ['K', 'L', 'M', 'N', 'O'],
        'description': 'Letters K-O'
    },
    'ALPHA4': {
        'name': 'Alphabet Group 4',
        'type': 'alphabet',
        'classes': ['P', 'Q', 'R', 'S', 'T'],
        'description': 'Letters P-T'
    },
    'ALPHA5': {
        'name': 'Alphabet Group 5',
        'type': 'alphabet',
        'classes': ['U', 'V', 'W', 'X', 'Y'],
        'description': 'Letters U-Y'
    },
    'ALPHA6': {
        'name': 'Alphabet Group 6',
        'type': 'alphabet',
        'classes': ['Z', 'space', 'nothing', 'del', 'A'],
        'description': 'Letters Z + Special (space, nothing, del)'
    },
    
    # SIGN GROUPS (7 groups - NO similar signs together)
    'SIGN1': {
        'name': 'Sign Group 1 - Mixed Actions & Objects',
        'type': 'sign',
        'classes': ['before', 'dance', 'dog', 'need', 'school'],
        'description': 'Timing, action, animal, request, place - ALL DIFFERENT'
    },
    'SIGN2': {
        'name': 'Sign Group 2 - Mixed Concepts & Actions',
        'type': 'sign',
        'classes': ['computer', 'cook', 'family', 'fine', 'later'],
        'description': 'Tech, action, family, expression, timing - ALL DIFFERENT'
    },
    'SIGN3': {
        'name': 'Sign Group 3 - Mixed Expressions & Movement',
        'type': 'sign',
        'classes': ['cool', 'drink', 'enjoy', 'go', 'man'],
        'description': 'Expression, consumption, emotion, movement, person - cool & enjoy SEPARATED'
    },
    'SIGN4': {
        'name': 'Sign Group 4 - Mixed Consumption & Relations',
        'type': 'sign',
        'classes': ['eat', 'finish', 'help', 'how', 'mother'],
        'description': 'Consumption, action, request, question, family - help SEPARATED from want'
    },
    'SIGN5': {
        'name': 'Sign Group 5 - Mixed Emotions & Negation',
        'type': 'sign',
        'classes': ['give', 'like', 'meet', 'no', 'play'],
        'description': 'Action, emotion, social, negation, activity - like SEPARATED from enjoy, no from yes'
    },
    'SIGN6': {
        'name': 'Sign Group 6 - Mixed Communication & Movement',
        'type': 'sign',
        'classes': ['now', 'son', 'tell', 'walk', 'woman'],
        'description': 'Timing, family, communication, movement, person - walk SEPARATED from go, woman from man'
    },
    'SIGN7': {
        'name': 'Sign Group 7 - Requests, Questions & Concepts',
        'type': 'sign',
        'classes': ['want', 'what', 'who', 'work', 'wrong', 'yes'],
        'description': 'Request, questions, place, negation, affirmation - want SEPARATED from help, yes from no'
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# SEARCH DATABASE
# ═══════════════════════════════════════════════════════════════════════════

SIGNS_DB = {}
ALPHABETS_DB = {}

for group_id, group_info in GROUPS.items():
    for class_name in group_info['classes']:
        search_entry = {
            'group_id': group_id,
            'group_name': group_info['name'],
            'type': group_info['type'],
            'class_name': class_name
        }
        
        if group_info['type'] == 'alphabet':
            ALPHABETS_DB[class_name] = search_entry
        else:
            SIGNS_DB[class_name] = search_entry

ALL_CLASSES_DB = {**SIGNS_DB, **ALPHABETS_DB}

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TRAINING_CONFIG = {
    'samples_per_class': 400,
    'raw_samples_to_collect': 100,
    'augmentation_factor': 4,
    
    'random_forest': {
        'n_estimators': 300,
        'max_depth': 20,
        'min_samples_split': 2,
        'random_state': 42,
    },
    
    'xgboost': {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
    },
    
    'meta_learner': {
        'model_type': 'logistic_regression',
        'max_iter': 1000,
        'random_state': 42,
    },
    
    'cv_folds': 5,
    'stratified': True,
}

# ═══════════════════════════════════════════════════════════════════════════
# FLASK CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

APP_CONFIG = {
    'host': '127.0.0.1',
    'port': 5000,
    'debug': False,
    'frame_width': 640,
    'frame_height': 480,
    'fps_target': 15,
    'jpeg_quality': 60,
    'confidence_threshold': 0.6,
    'smooth_buffer': 5,
}

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_group_by_class(class_name):
    """Get group information for a specific class"""
    if class_name in ALL_CLASSES_DB:
        group_id = ALL_CLASSES_DB[class_name]['group_id']
        return GROUPS[group_id]
    return None

def get_classes_by_group(group_id):
    """Get all classes in a specific group"""
    if group_id in GROUPS:
        return GROUPS[group_id]['classes']
    return None

def search_class(query):
    """Search for a class by name (case-insensitive)"""
    query_lower = query.lower()
    return ALL_CLASSES_DB.get(query_lower, None)

def get_all_groups():
    """Get all group information"""
    return GROUPS

def get_group_count():
    """Get total number of groups"""
    return len(GROUPS)

def get_total_classes():
    """Get total number of classes"""
    return len(ALL_CLASSES_DB)

# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

STATS = {
    'total_groups': len(GROUPS),
    'total_classes': len(ALL_CLASSES_DB),
    'sign_classes': len(SIGNS_DB),
    'alphabet_classes': len(ALPHABETS_DB),
    'total_samples_needed': len(ALL_CLASSES_DB) * 400,
    'total_raw_samples_needed': len(ALL_CLASSES_DB) * 100,
}

if __name__ == '__main__':
    print("=" * 70)
    print("IMPROVED ASL - CONFIGURATION (UPDATED)")
    print("=" * 70)
    print(f"\nTotal Groups: {STATS['total_groups']}")
    print(f"Total Classes: {STATS['total_classes']}")
    print(f"  - Sign Classes: {STATS['sign_classes']}")
    print(f"  - Alphabet Classes: {STATS['alphabet_classes']}")
    print(f"\nTotal Samples Needed: {STATS['total_samples_needed']:,}")
    print(f"Total Raw Samples Needed: {STATS['total_raw_samples_needed']:,}")
    
    print("\n" + "=" * 70)
    print("GROUPS:")
    print("=" * 70)
    for group_id, group_info in GROUPS.items():
        print(f"\n{group_id}: {group_info['name']}")
        print(f"  Type: {group_info['type']}")
        print(f"  Classes: {', '.join(group_info['classes'])}")
        print(f"  Description: {group_info['description']}")
    
    print("\n" + "=" * 70)
    print("PATHS:")
    print("=" * 70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Signs Dataset: {SIGNS_FRAMES_DIR}")
    print(f"Alphabets Dataset: {ALPHABETS_DIR}")
    print(f"Keypoints Output: {KEYPOINTS_DIR}")
    print(f"Models Output: {MODELS_DIR}")
    print(f"Logs Output: {LOGS_DIR}")
    print("=" * 70)