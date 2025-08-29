# prompts for models 
ZERO_SHOT_EN = """You are given two sentences that both contain the same target word.
Your task is to determine whether the word has the same meaning in both sentences.
If the meaning is the same, answer "True".
If the meaning is different, answer "False".
Respond with only "True" or "False".

Sentence 1: {sentence1}
Sentence 2: {sentence2}
Word: {target_word}
Answer:
"""

FEW_SHOT_EN = """You are given two sentences that both contain the same target word.
Your task is to determine whether the word has the same meaning in both sentences.
If the meaning is the same, answer "True".
If the meaning is different, answer "False".
Respond with only "True" or "False".

Examples:

Sentence 1: Ол үстелдің басында отырды. 
Sentence 2: Оның басы ауырды.
Word: бас
Answer: False

Sentence 1: Ол ағылшын тілін үйреніп жүр.
Sentence 2: Қазақ тілі - мемлекеттік тіл.
Word: тіл
Answer: True

Now your task:
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Word: {target_word}
Answer: 
"""

ZERO_SHOT_KK = """Саған бірдей мақсатты сөзді қамтитын екі сөйлем берілген.
Сенің тапсырмаң – осы сөздің екі сөйлемде де бірдей мағынада қолданылған-қолданылмағанын анықтау.
Егер мағынасы бірдей болса, "True" деп жауап бер.
Егер мағынасы әртүрлі болса, "False" деп жауап бер.
Тек "True" немесе "False" деп жауап бер.

Сөйлем 1: {sentence1}
Сөйлем 2: {sentence2}
Сөз: {target_word}
Жауап:
"""

FEW_SHOT_KK = """Саған бірдей мақсатты сөзді қамтитын екі сөйлем берілген.  
Сенің тапсырмаң – осы сөздің екі сөйлемде де бірдей мағынада қолданылған-қолданылмағанын анықтау.  
Егер мағынасы бірдей болса, "True" деп жауап бер.  
Егер мағынасы әртүрлі болса, "False" деп жауап бер.    
Тек "True" немесе "False" деп жауап бер.

Мысалдар:

Сөйлем 1: Ол үстелдің басында отырды.
Сөйлем 2: Оның басы ауырды.
Сөз: бас
Жауап: False

Сөйлем 1: Ол ағылшын тілін үйреніп жүр.  
Сөйлем 2: Қазақ тілі - мемлекеттік тіл.
Сөз: тіл
Жауап: True

Енді сенің тапсырмаң:
Сөйлем 1: {sentence1}  
Сөйлем 2: {sentence2}  
Сөз: {target_word}
Жауап:
"""

PROMPTS = {
    ("zero", "en"): ZERO_SHOT_EN,
    ("few", "en"): FEW_SHOT_EN,
    ("zero", "kk"): ZERO_SHOT_KK,
    ("few", "kk"): FEW_SHOT_KK,
}
