"""
IMPROVED ASL - Configuration File
==================================
Defines all classes, groups, paths, and search database
Total classes updated with exclusions
Organized into focused groups plus auto-scan universal groups
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
SIGNS_FRAMES_DIR = DATASET_DIR / "frames_20"
ALPHABETS_DIR = DATASET_DIR / "archive" / "asl_alphabet_train" / "asl_alphabet_train"
KEYPOINTS_DIR = BASE_DIR / "keypoints_extracted"
MODELS_DIR = BASE_DIR / "Models"
LOGS_DIR = BASE_DIR / "logs"

SIGN_CLASSES = {
    0: 'before', 1: 'computer', 2: 'dance', 3: 'dog', 4: 'drink', 5: 'eat', 6: 'enjoy',
    7: 'fine', 8: 'finish', 9: 'give', 10: 'go', 11: 'help', 12: 'how', 13: 'later', 14: 'like',
    15: 'meet', 16: 'need', 17: 'no', 18: 'now', 19: 'play', 20: 'school',
    21: 'tell', 22: 'want', 23: 'what', 24: 'who', 25: 'work', 26: 'wrong', 27: 'yes',
}

ALPHABET_CLASSES = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'space', 27: 'nothing', 28: 'del',
}

ALL_CLASS_NAMES = list(SIGN_CLASSES.values()) + list(ALPHABET_CLASSES.values())
CLASS_IDS = {
    **{name: class_id for class_id, name in SIGN_CLASSES.items()},
    **{name: 28 + class_id for class_id, name in ALPHABET_CLASSES.items()},
}

GROUPS = {
    'ALPHA1': {'name': 'Alphabet Group 1', 'type': 'alphabet', 'classes': ['A', 'B', 'C', 'D', 'E'], 'description': 'Letters A-E'},
    'ALPHA2': {'name': 'Alphabet Group 2', 'type': 'alphabet', 'classes': ['F', 'G', 'H', 'I', 'J'], 'description': 'Letters F-J'},
    'ALPHA3': {'name': 'Alphabet Group 3', 'type': 'alphabet', 'classes': ['K', 'L', 'M', 'N', 'O'], 'description': 'Letters K-O'},
    'ALPHA4': {'name': 'Alphabet Group 4', 'type': 'alphabet', 'classes': ['P', 'Q', 'R', 'S', 'T'], 'description': 'Letters P-T'},
    'ALPHA5': {'name': 'Alphabet Group 5', 'type': 'alphabet', 'classes': ['U', 'V', 'W', 'X', 'Y'], 'description': 'Letters U-Y'},
    'ALPHA6': {'name': 'Alphabet Group 6', 'type': 'alphabet', 'classes': ['Z', 'space', 'nothing', 'del', 'A'], 'description': 'Letters Z + Special'},
    'SIGN1': {'name': 'Sign Group 1', 'type': 'sign', 'classes': ['before', 'dance', 'dog', 'need', 'school'], 'description': 'Timing, action, animal, request, place - ALL DIFFERENT'},
    'SIGN2': {'name': 'Sign Group 2', 'type': 'sign', 'classes': ['computer', 'fine', 'later'], 'description': 'Tech, expression, timing - ALL DIFFERENT'},
    'SIGN3': {'name': 'Sign Group 3', 'type': 'sign', 'classes': ['drink', 'enjoy', 'go'], 'description': 'Consumption, emotion, movement'},
    'SIGN4': {'name': 'Sign Group 4', 'type': 'sign', 'classes': ['eat', 'finish', 'help', 'how'], 'description': 'Consumption, action, request, question'},
    'SIGN5': {'name': 'Sign Group 5', 'type': 'sign', 'classes': ['give', 'like', 'meet', 'no', 'play'], 'description': 'Action, emotion, social, negation, activity'},
    'SIGN6': {'name': 'Sign Group 6', 'type': 'sign', 'classes': ['now', 'tell'], 'description': 'Timing, communication'},
    'SIGN7': {'name': 'Sign Group 7', 'type': 'sign', 'classes': ['want', 'what', 'who', 'work', 'wrong', 'yes'], 'description': 'Request, questions, place, negation, affirmation'},
    'ALPHA_ALL': {'name': 'Alphabet Auto-Scan', 'type': 'alphabet_auto', 'classes': list(ALPHABET_CLASSES.values()), 'description': 'All alphabet classes'},
    'SIGN_ALL': {'name': 'Word Auto-Scan', 'type': 'sign_auto', 'classes': list(SIGN_CLASSES.values()), 'description': 'All sign classes'},
}

SIGNS_DB = {}
ALPHABETS_DB = {}

for group_id, group_info in GROUPS.items():
    if group_info['type'] in ('universal', 'alphabet_auto', 'sign_auto'):
        continue
    for class_name in group_info['classes']:
        search_entry = {'group_id': group_id, 'group_name': group_info['name'], 'type': group_info['type'], 'class_name': class_name}
        if group_info['type'] == 'alphabet':
            ALPHABETS_DB[class_name] = search_entry
        else:
            SIGNS_DB[class_name] = search_entry

ALL_CLASSES_DB = {**SIGNS_DB, **ALPHABETS_DB}

TRAINING_CONFIG = {
    'samples_per_class': 400, 'raw_samples_to_collect': 100, 'augmentation_factor': 4,
    'random_forest': {'n_estimators': 300, 'max_depth': 20, 'random_state': 42},
    'xgboost': {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42},
    'meta_learner': {'model_type': 'logistic_regression', 'max_iter': 1000, 'random_state': 42},
    'cv_folds': 5, 'stratified': True,
}

APP_CONFIG = {
    'host': '127.0.0.1', 'port': 5000, 'debug': False,
    'frame_width': 640, 'frame_height': 480, 'fps_target': 15, 'jpeg_quality': 60,
    'confidence_threshold': 0.6, 'smooth_buffer': 5,
}

def get_group_by_class(class_name):
    if class_name in ALL_CLASSES_DB:
        group_id = ALL_CLASSES_DB[class_name]['group_id']
        return GROUPS[group_id]
    return None

def get_classes_by_group(group_id):
    if group_id in GROUPS:
        return GROUPS[group_id]['classes']
    return None

def search_class(query):
    query_lower = query.lower()
    return ALL_CLASSES_DB.get(query_lower, None)

def get_all_groups():
    return GROUPS

def get_group_count():
    return len(GROUPS)

def get_total_classes():
    return len(ALL_CLASSES_DB)

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
    print("IMPROVED ASL - CONFIGURATION")
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