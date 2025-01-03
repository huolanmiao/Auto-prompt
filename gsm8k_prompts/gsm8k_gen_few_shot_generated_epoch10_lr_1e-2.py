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
                dict(role='HUMAN', prompt="Question: Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?\nAnswer:"),
                dict(role='BOT', prompt="Ken started with a box that weighed 2 pounds after adding jelly beans. Then, he added enough brownies to triple the weight, so the weight became 2 pounds * 3 = 6 pounds. Next, he added another 2 pounds of jelly beans, bringing the total weight to 6 pounds + 2 pounds = 8 pounds. Finally, he added enough gummy worms to double the weight again, so the final weight became 8 pounds * 2 = 16 pounds. The answer is 16."),
                dict(role='HUMAN', prompt="Question: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\nAnswer:"),
                dict(role='BOT', prompt="Joy can read 8 pages in 20 minutes. To find out how many minutes it takes her to read 1 page, we divide 20 minutes by 8 pages: 20 / 8 = 2.5 minutes per page.\nNow, to read 120 pages, we multiply the time it takes to read one page by the total number of pages: 2.5 minutes/page * 120 pages = 300 minutes.\nTo convert minutes to hours, we divide by 60: 300 minutes / 60 = 5 hours.\nSo, it will take Joy 5 hours to read 120 pages. The answer is 5."),
                dict(role='HUMAN', prompt="Question: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name?\nAnswer:"),
                dict(role='BOT', prompt="Jamie's last name is Grey, which has 4 letters. If Bobbie took 2 letters off her last name, she would have a last name twice the length of Jamie's, which means Bobbie's last name would be 4 * 2 = 8 letters long after removing 2 letters. Therefore, Bobbie's original last name must have been 8 + 2 = 10 letters long. Since Samantha's last name has 3 fewer letters than Bobbie's, Samantha's last name must have 10 - 3 = 7 letters. The answer is 7."),
                dict(role='HUMAN', prompt="Question: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name?\nAnswer:"),
                dict(role='BOT', prompt="Jamie's last name is Grey, which has 4 letters. If Bobbie took 2 letters off her last name, she would have a last name twice the length of Jamie's, which means Bobbie's last name would be 4 * 2 = 8 letters long after removing 2 letters. Therefore, Bobbie's original last name must have been 8 + 2 = 10 letters long. Since Samantha's last name has 3 fewer letters than Bobbie's, Samantha's last name must have 10 - 3 = 7 letters. The answer is 7."),
                dict(role='HUMAN', prompt="Question: Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients?\nAnswer:"),
                dict(role='BOT', prompt="Jasper buys 2 pounds of cheddar cheese for $10. The cream cheese costs half the price of the cheddar cheese, so it costs $10 / 2 = $5. The pack of cold cuts costs twice the price of the cheddar cheese, so it costs $10 * 2 = $20. To find the total cost, we add the cost of the cheddar cheese, cream cheese, and cold cuts: $10 + $5 + $20 = $35. The answer is 35."),
                dict(role='HUMAN', prompt="Question: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?\nAnswer:"),
                dict(role='BOT', prompt="Let's denote the number of people on the ship in the first hundred years as \( x \). Since each subsequent ship has twice as many people as the last, the number of people on the ships in the second and third hundred years would be \( 2x \) and \( 4x \) respectively.\nOver three hundred years, the total number of people consumed is:\n\[ x + 2x + 4x = 7x \]\nWe know that the total number of people consumed is 847, so:\n\[ 7x = 847 \]\nSolving for \( x \):\n\[ x = \frac{847}{7} = 121 \]\nSo, there were 121 people on the ship the monster ate in the first hundred years. The answer is 121."),
                dict(role='HUMAN', prompt="Question: Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?\nAnswer:"),
                dict(role='BOT', prompt="Ken started with a box that weighed 2 pounds after adding jelly beans. Then, he added enough brownies to triple the weight, so the weight became 2 pounds * 3 = 6 pounds. Next, he added another 2 pounds of jelly beans, bringing the total weight to 6 pounds + 2 pounds = 8 pounds. Finally, he added enough gummy worms to double the weight again, so the final weight became 8 pounds * 2 = 16 pounds. The answer is 16."),
                dict(role='HUMAN', prompt="Question: Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?\nAnswer:"),
                dict(role='BOT', prompt="Alexis started with a budget of $200. She spent $30 on a shirt, $46 on pants, $38 on a coat, $11 on socks, and $18 on a belt. The total amount she spent on these items is $30 + $46 + $38 + $11 + $18 = $143. After these purchases, she had $200 - $143 = $57 left. However, she has $16 left after purchasing the shoes. Therefore, the amount she spent on the shoes must be $57 - $16 = $41. The answer is 41."),
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
