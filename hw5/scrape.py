import requests
from bs4 import BeautifulSoup

def scrape_data():
    url = "http://www.aos.wisc.edu/~sco/lakes/Mendota-ice.html"
    r = requests.get(url)
    contents = r.text
    bs_obj = BeautifulSoup(contents, "html.parser")
    
    td_elements = bs_obj.find_all("td")

    td_useful = []
    td_useful.append(td_elements[12])
    td_useful.append(td_elements[17])
    td_useful.append(td_elements[30])
    td_useful.append(td_elements[35])
    td_useful.append(td_elements[48])
    td_useful.append(td_elements[53])
    td_useful.append(td_elements[66])
    td_useful.append(td_elements[71])

    td_data = [td.get_text().replace('\xa0',' ').strip().split("\n") for td in td_useful]
    td_data[0] = td_data[0][3:]
    last_data = td_data[-1].pop(-1)
    td_data[-1].extend(last_data.split(' '))
    td_data[-1].pop(3)
    
    fix_2nd = td_data[-1][1].strip().split(' ')
    td_data[-1][1] = fix_2nd[0]
    td_data[-1].insert(2, fix_2nd[1])

    with open("hw5.csv", "w") as f:
        f.write("year,days\n")
        year = 1855
        for td_element in td_data:
            for day in td_element:
                day = day.replace(' ','')
                if day.startswith("-"):
                    continue
                if year == 2021:
                    break
                f.write(f'{year},{day}\n')
                year += 1

scrape_data()