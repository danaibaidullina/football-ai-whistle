# ===== Блок: ошибки судей с карточками, Premier League 2024/25 =====
# Данные: data/card_errors_2425_all.csv (собраны вручную из списков ESPN по оценкам панели
# Key Match Incidents Premier League). Колонка favored = кому ошибка выгодна: home / away.
import pandas as pd
from scipy.stats import binomtest

errs = pd.read_csv("data/card_errors_2425_all.csv")

def split(df, label):
    n = len(df)
    k = int((df.favored == "home").sum())
    p = binomtest(k, n, 0.5).pvalue          # точный тест, нулевая гипотеза: 50 на 50
    ci = binomtest(k, n, 0.5).proportion_ci(confidence_level=0.95)
    print(f"{label:<14} всего {n:>2} | в пользу хозяев {k:>2} | в пользу гостей {n-k:>2} | "
          f"доля хозяев {k/n:.2f} [{ci.low:.2f}; {ci.high:.2f}] | p = {p:.3f}")

split(errs, "Все вместе")
split(errs[errs.kind == "second_yellow"], "Вторые жёлтые")
split(errs[errs.kind == "red"], "Красные")

# Простой график
ax = (errs.groupby(["kind", "favored"]).size().unstack(fill_value=0)
          .rename(index={"second_yellow": "Вторые жёлтые", "red": "Красные"},
                  columns={"home": "в пользу хозяев", "away": "в пользу гостей"})
          .plot(kind="bar", rot=0, color=["#E0B458", "#6FD3A5"]))
ax.set_ylabel("число ошибок"); ax.set_title("Ошибки с карточками, АПЛ 2024/25")
