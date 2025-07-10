from bs4 import BeautifulSoup
import requests
import time
from random import random
import typer
import numpy as np
import polars as pl

app = typer.Typer()

@app.command()
def main(round: int = typer.Option(1, "--round", "-r", help="The round to scrape")):
    print(f"🎯 로또 당첨 번호 수집 시작 (총 {round}회차)")
    winning_numbers = []
    df = pl.read_csv("./numbers.csv")
    for v in df.iter_rows(named=True):
        numbers = [int(v["1"]), int(v["2"]), int(v["3"]), int(v["4"]), int(v["5"]), int(v["6"])]
    # for r in range(1, round+1):
        # print(f"🎯 회차 {r} 수집 시작")
        # page = get_winning_numbers_page(r)
        # print(page)
        # numbers = parse_winning_numbers(page)
        print(numbers)
        winning_numbers.append(numbers)
    one_hots = to_one_hot(winning_numbers)
    np.save("app/data/lotto_one_hot.npy", one_hots)
    print(f"✅ 수집 완료. 총 {len(one_hots)}개 회차 저장됨.")

def get_winning_numbers(round: int) -> list[int]:
    df = pl.read_csv("numbers.csv")
    for v in df.iter_rows(named=True):
        d = [int(v["1"]), int(v["2"]), int(v["3"]), int(v["4"]), int(v["5"]), int(v["6"])]
        
    return numbers

def get_winning_numbers_page(round: int) -> bytes:
    url = "https://dhlottery.co.kr/gameResult.do"
    params = {"method": "byWin"}
    data={
        "drwNo": round,
        "hdrwComb": 1,
        "dwrNoList": round
    }
    response = requests.post(url, params=params, json=data, 
                             proxies={'http':'http://50.114.15.36:6021','https':'http://104.239.105.150:6680'})
    return response.content

def parse_winning_numbers(page: bytes) -> list[int]:
    soup = BeautifulSoup(page, "lxml")
    winning_numbers = [int(span.text) for span in soup.select("div.win_result > div.nums")[0].find_all("span")]
    return winning_numbers

def to_one_hot(numbers: list[list[int]], num_range: int = 45) -> np.ndarray:
    one_hots = []
    for draw in numbers:
        vec = [0] * num_range
        for num in draw:
            vec[num-1] = 1
        one_hots.append(vec)
    return np.array(one_hots)

if __name__ == "__main__":
    app()