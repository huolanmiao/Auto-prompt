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
                dict(role='HUMAN', prompt="Question: It takes Roque two hours to walk to work and one hour to ride his bike to work. Roque walks to and from work three times a week and rides his bike to and from work twice a week. How many hours in total does he take to get to and from work a week with walking and biking?\nAnswer:"),
                dict(role='BOT', prompt="Roque walks to and from work three times a week. Since it takes him two hours to walk to work and two hours to walk back, each walk trip takes 2 + 2 = 4 hours. Over three trips, he spends 3 * 4 = 12 hours walking.\nRoque rides his bike to and from work twice a week. Since it takes him one hour to ride to work and one hour to ride back, each bike trip takes 1 + 1 = 2 hours. Over two trips, he spends 2 * 2 = 4 hours biking.\nIn total, Roque spends 12 hours walking and 4 hours biking, which adds up to 12 + 4 = 16 hours. The answer is 16.\n"),
                dict(role='HUMAN', prompt="Question: It takes Roque two hours to walk to work and one hour to ride his bike to work. Roque walks to and from work three times a week and rides his bike to and from work twice a week. How many hours in total does he take to get to and from work a week with walking and biking?\nAnswer:"),
                dict(role='BOT', prompt="Roque walks to and from work three times a week. Since it takes him two hours to walk to work and two hours to walk back, each walk trip takes 2 + 2 = 4 hours. Over three trips, he spends 3 * 4 = 12 hours walking.\nRoque rides his bike to and from work twice a week. Since it takes him one hour to ride to work and one hour to ride back, each bike trip takes 1 + 1 = 2 hours. Over two trips, he spends 2 * 2 = 4 hours biking.\nIn total, Roque spends 12 hours walking and 4 hours biking, which adds up to 12 + 4 = 16 hours. The answer is 16.\n"),
                dict(role='HUMAN', prompt="Question: Bella bought stamps at the post office. Some of the stamps had a snowflake design, some had a truck design, and some had a rose design. Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stamps, and 13 fewer rose stamps than truck stamps. How many stamps did Bella buy in all?\nAnswer:"),
                dict(role='BOT', prompt="Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stamps, so she bought 11 + 9 = 20 truck stamps. She bought 13 fewer rose stamps than truck stamps, so she bought 20 - 13 = 7 rose stamps. In total, Bella bought 11 + 20 + 7 = 38 stamps. The answer is 38."),
                dict(role='HUMAN', prompt="Question: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?\nAnswer:"),
                dict(role='BOT', prompt="Let's denote the number of people on the ship in the first hundred years as \( x \). Since each subsequent ship has twice as many people as the last, the number of people on the ships in the second and third hundred years would be \( 2x \) and \( 4x \) respectively.\nOver three hundred years, the total number of people consumed is:\n\[ x + 2x + 4x = 7x \]\nWe know that the total number of people consumed is 847, so:\n\[ 7x = 847 \]\nSolving for \( x \):\n\[ x = \frac{847}{7} = 121 \]\nSo, there were 121 people on the ship the monster ate in the first hundred years. The answer is 121."),
                dict(role='HUMAN', prompt="Question: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\nAnswer:"),
                dict(role='BOT', prompt="Joy can read 8 pages in 20 minutes. To find out how many minutes it takes her to read 1 page, we divide 20 minutes by 8 pages: 20 / 8 = 2.5 minutes per page.\nNow, to read 120 pages, we multiply the time it takes to read one page by the total number of pages: 2.5 minutes/page * 120 pages = 300 minutes.\nTo convert minutes to hours, we divide by 60: 300 minutes / 60 = 5 hours.\nSo, it will take Joy 5 hours to read 120 pages. The answer is 5."),
                dict(role='HUMAN', prompt="Question: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\nAnswer:"),
                dict(role='BOT', prompt="Joy can read 8 pages in 20 minutes. To find out how many minutes it takes her to read 1 page, we divide 20 minutes by 8 pages: 20 / 8 = 2.5 minutes per page.\nNow, to read 120 pages, we multiply the time it takes to read one page by the total number of pages: 2.5 minutes/page * 120 pages = 300 minutes.\nTo convert minutes to hours, we divide by 60: 300 minutes / 60 = 5 hours.\nSo, it will take Joy 5 hours to read 120 pages. The answer is 5."),
                dict(role='HUMAN', prompt="Question: Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?\nAnswer:"),
                dict(role='BOT', prompt="First, we need to find out how many coconut trees Randy has. He has 5 less than half as many coconut trees as mango trees. Half of the mango trees is 60 / 2 = 30. So, he has 30 - 5 = 25 coconut trees.\nNow, we add the number of mango trees and coconut trees together: 60 mango trees + 25 coconut trees = 85 trees. The answer is 85.\n"),
                dict(role='HUMAN', prompt="Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:"),
                dict(role='BOT', prompt="Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips. To find the total number of clips sold in April and May, we add the two amounts together: 48 + 24 = 72 clips. The answer is 72."),
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
