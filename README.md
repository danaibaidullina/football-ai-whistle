# Is the whistle fair? Referees, crowds and refereeing errors in football

HW5 "AI Football Intelligence" (Cognitive technology + DSS).
Team: Danagul Ibaidullina, Alyona Abaidullina, Diana Slyamkhanova.

## Question
When the stadium is full, do away teams get more yellow cards than home teams **for the same number of fouls**?
In 2020 many matches were played without spectators (COVID), which gives a natural experiment.
As a second layer, the project looks at documented refereeing errors involving cards in the Premier League 2024/25 and ask whom they favoured.

**User / decision:** the refereeing department of a league: where do decisions depend on the atmosphere in the stadium, and what should be monitored.

## Data
| Source | What | Used for |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/data.php) | match results, fouls (HF/AF), yellow cards (HY/AY), shots, referee (Premier League only) | main analysis, 11 leagues, 2017-2025 |
| ESPN lists based on the Premier League Key Match Incidents panel (2024/25) | 27 refereeing errors with cards | `data/card_errors_2425_all.csv` (collected by hand) |

Matches are labelled by date: with crowd (before 2020-03-01), pause (excluded), empty stadiums (2019/20 restart), partly empty 2020/21 (excluded from the main comparison), crowd back (2021/22-2022/23).

## Method
Metric: yellow cards per foul, gap = away minus home. Bootstrap confidence intervals, Poisson regression with offset log(fouls) and match-clustered errors, team fixed effects, placebo permutation test, leave-one-league-out, a gradient-boosting model trained on matches with crowd, a pre-match card forecast (time split, rolling form with shift(1) to avoid leakage), an LLM assistant that calls the forecast tools and has its numbers checked by code.

## Results
| Period | Gap, cards per foul (away - home) |
|---|---|
| Crowd (before COVID) | +0.0188 [+0.0164; +0.0212] |
| Empty stadiums | -0.0063 [-0.0136; +0.0010] |
| Crowd back | +0.0189 |
| Crowd minus empty | +0.0251 [+0.0173; +0.0329] |

About 2 more cards per 100 fouls for the away team when the stadium is full, and no gap when it is empty.

- Placebo (random "empty" labels): range [-0.0081; +0.0082] vs observed +0.0251 (0 of 2000).
- Leave-one-league-out: +0.0225 ... +0.0274. Positive in 10 of 11 leagues.
- Regression: home teams get fewer cards per foul with a crowd (x0.925); this discount disappears without spectators.
- ML check (5 leagues): a model trained on crowd matches mis-predicts empty matches by -0.32 cards per team-match.
- **Negative result:** "weak teams are punished more" does not hold once shots are controlled (coefficient x1.0000, p = 0.997).
- **Forecast:** MAE 1.0905 vs 1.1190 (league mean) vs 1.1541 (recent form) on unseen seasons. A small gain. Adding the referee did not improve accuracy on 754 matches (+0.0049 [-0.0200; +0.0292]).

### Refereeing errors with cards, Premier League 2024/25 (27 cases)
| Type | Total | Favoured home | Favoured away | p (exact binomial, 50/50) |
|---|---|---|---|---|
| Second yellow | 15 | 11 | 4 | 0.12 |
| Red card | 12 | 7 | 5 | 0.77 |
| All | 27 | 18 | 9 | 0.12 |

The pattern is in line with the hypothesis, but the sample is small and the imbalance could be chance.
Single yellow cards are not reviewed by the panel, so "a yellow that should not have been shown" cannot be tested.

## Limitations
- Causality is not proven: empty matches were also the restart after the pause (compressed schedule, five substitutions).
- Period is defined by date, not checked per country.
- Referees are only in the Premier League data; hands, offsides and VAR decisions are not in open data.
- Several p-values are around 0.02-0.08 and many tests were run; the argument relies on the same picture across methods.
- Error data: one season, one league, collected by hand; "favoured home/away" assumes 50/50 chances, which is rough.
- The LLM check catches invented numbers but not misplaced ones.

## AI usage
Claude (LLM) was used as an assistant as needed.

## Run
```
pip install -r requirements.txt
```
Run the notebook top to bottom. The LLM assistant needs `ANTHROPIC_API_KEY` stored in Colab Secrets (never in code). The refereeing-errors block is in `errors_block.py` and reads `data/card_errors_2425_all.csv`.
