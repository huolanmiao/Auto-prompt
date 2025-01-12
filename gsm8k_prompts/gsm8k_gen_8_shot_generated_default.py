from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator


gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:"),
                dict(role='BOT', prompt='Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips. To find the total number of clips sold in April and May, we add the two amounts together: 48 + 24 = 72 clips. The answer is 72.'),
                dict(role='HUMAN', prompt="Question: Each bird eats 12 beetles per day, each snake eats 3 birds per day, and each jaguar eats 5 snakes per day. If there are 6 jaguars in a forest, how many beetles are eaten each day?\nAnswer:"),
                dict(role='BOT', prompt='Each jaguar eats 5 snakes per day. There are 6 jaguars, so they eat a total of 6 * 5 = 30 snakes per day.\nEach snake eats 3 birds per day. Since 30 snakes are eaten by the jaguars, the snakes would have eaten 30 * 3 = 90 birds per day.\nEach bird eats 12 beetles per day. Since 90 birds are eaten by the snakes, the birds would have eaten 90 * 12 = 1080 beetles per day.\nTherefore, the total number of beetles eaten each day is 1080. The answer is 1080.'),
                dict(role='HUMAN', prompt="Question: Tim rides his bike back and forth to work for each of his 5 workdays.  His work is 20 miles away.  He also goes for a weekend bike ride of 200 miles.    If he can bike at 25 mph how much time does he spend biking a week?\nAnswer:"),
                dict(role='BOT', prompt="First, let's calculate the total distance Tim bikes during the workdays. He rides back and forth, so he travels 2 * 20 miles each day. Over 5 workdays, he travels 5 * (2 * 20) = 5 * 40 = 200 miles.\nHe also goes for a weekend bike ride of 200 miles. So, the total distance he bikes in a week is 200 (workdays) + 200 (weekend) = 400 miles.\nNow, we need to calculate the time he spends biking. Since he bikes at 25 mph, the time he spends biking is the total distance divided by his speed: 400 miles / 25 mph = 16 hours.\nThe answer is 16."),
                dict(role='HUMAN', prompt="Question: In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.\nAnswer:"),
                dict(role='BOT', prompt="First, let's calculate the total number of hard hats initially:\n26 pink + 15 green + 24 yellow = 65 hard hats.\nCarl takes away 4 pink hard hats, so there are now 26 - 4 = 22 pink hard hats left.\nJohn takes away 6 pink hard hats, so there are now 22 - 6 = 16 pink hard hats left.\nJohn also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 * 6 = 12 green hard hats. So there are now 15 - 12 = 3 green hard hats left.\nNow, let's add up the remaining hard hats:\n16 pink + 3 green + 24 yellow = 43 hard hats.\nThe total number of hard hats that remained in the truck is 43. The answer is 43."),
                dict(role='HUMAN', prompt="Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer:"),
                dict(role='BOT', prompt="Betty has half of the money she needs, which is $100 / 2 = $50. Her parents gave her $15, and her grandparents gave her twice as much as her parents, which is 2 * $15 = $30. So, in total, she received $15 + $30 = $45 from her parents and grandparents. Now, she has $50 + $45 = $95. To buy the wallet, she needs $100, so she still needs $100 - $95 = $5 more. The answer is 5."),
                dict(role='HUMAN', prompt="Question: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\nAnswer:"),
                dict(role='BOT', prompt="Joy can read 8 pages in 20 minutes. To find out how many minutes it takes her to read 1 page, we divide 20 minutes by 8 pages: 20 / 8 = 2.5 minutes per page.\nNow, to read 120 pages, we multiply the time it takes to read one page by the total number of pages: 2.5 minutes/page * 120 pages = 300 minutes.\nTo convert minutes to hours, we divide by 60: 300 minutes / 60 = 5 hours.\nSo, it will take Joy 5 hours to read 120 pages. The answer is 5."),
                dict(role='HUMAN', prompt="Question: Each bird eats 12 beetles per day, each snake eats 3 birds per day, and each jaguar eats 5 snakes per day. If there are 6 jaguars in a forest, how many beetles are eaten each day?\nAnswer:"),
                dict(role='BOT', prompt='Each jaguar eats 5 snakes per day. There are 6 jaguars, so they eat a total of 6 * 5 = 30 snakes per day.\nEach snake eats 3 birds per day. Since 30 snakes are eaten by the jaguars, the snakes would have eaten 30 * 3 = 90 birds per day.\nEach bird eats 12 beetles per day. Since 90 birds are eaten by the snakes, the birds would have eaten 90 * 12 = 1080 beetles per day.\nTherefore, the total number of beetles eaten each day is 1080. The answer is 1080.'),
                dict(role='HUMAN', prompt="Question: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\nAnswer:"),
                dict(role='BOT', prompt="Joy can read 8 pages in 20 minutes. To find out how many minutes it takes her to read 1 page, we divide 20 minutes by 8 pages: 20 / 8 = 2.5 minutes per page.\nNow, to read 120 pages, we multiply the time it takes to read one page by the total number of pages: 2.5 minutes/page * 120 pages = 300 minutes.\nTo convert minutes to hours, we divide by 60: 300 minutes / 60 = 5 hours.\nSo, it will take Joy 5 hours to read 120 pages. The answer is 5."),
                dict(role='HUMAN', prompt="Question: {question}\nAnswer:"),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
