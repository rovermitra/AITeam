#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoverMitra – Realistic user generator (culture-aware names + rich travel onboarding, v2.0)

Exactly same schema and logic as your current script, only richer data and stricter geographic realism.

Key points preserved
- DO NOT include user_id or created_at
- Keep the same top-level signup fields
- Same feature structure and field names
- Lean profiles supported via LEAN_RATIO
- City always matches country, airport codes valid for that city, languages realistic

Improvements
- Much larger, curated country → [cities → airports] and language map
- Wider currency coverage
- Culture banks extended so names fit regions
- Phone numbers use country-aware codes
- Postal codes use simple country-aware patterns where possible
- ASCII-safe emails from unicode names
"""

import os
import json
import uuid
import random
import unicodedata
from datetime import datetime, date, timedelta

random.seed(77)

# =========================================================
# OUTPUT
# =========================================================
OUT_PATH   = "users/data/users_core.json"
N_USERS    = 10000
LEAN_RATIO = 0.20  # ~20% intentionally less complete

# =========================================================
# COUNTRY → CITIES/AIRPORTS/LANGS  (significantly expanded)
# A compact but realistic subset per country to keep file size reasonable.
# =========================================================
COUNTRIES = {
    # EUROPE CORE
    "Germany": {"langs":["de","en"],"cities":{
        "Berlin":["BER"],"Munich":["MUC"],"Hamburg":["HAM"],"Cologne":["CGN"],"Frankfurt":["FRA"],"Stuttgart":["STR"],"Düsseldorf":["DUS"],"Leipzig":["LEJ"]
    }},
    "Switzerland":{"langs":["de","fr","it","en"],"cities":{
        "Zurich":["ZRH"],"Geneva":["GVA"],"Basel":["BSL"],"Bern":["BRN"],"Lugano":["LUG"]
    }},
    "France":{"langs":["fr","en"],"cities":{
        "Paris":["CDG","ORY"],"Lyon":["LYS"],"Nice":["NCE"],"Marseille":["MRS"],"Toulouse":["TLS"],"Bordeaux":["BOD"]
    }},
    "UK":{"langs":["en"],"cities":{
        "London":["LHR","LGW","LCY","LTN","STN"],"Manchester":["MAN"],"Edinburgh":["EDI"],"Birmingham":["BHX"],"Glasgow":["GLA"],"Bristol":["BRS"]
    }},
    "Ireland":{"langs":["en","ga"],"cities":{
        "Dublin":["DUB"],"Cork":["ORK"],"Shannon":["SNN"]
    }},
    "Spain":{"langs":["es","en"],"cities":{
        "Madrid":["MAD"],"Barcelona":["BCN"],"Valencia":["VLC"],"Seville":["SVQ"],"Malaga":["AGP"],"Bilbao":["BIO"],"Palma de Mallorca":["PMI"]
    }},
    "Italy":{"langs":["it","en"],"cities":{
        "Rome":["FCO","CIA"],"Milan":["MXP","LIN","BGY"],"Florence":["FLR"],"Venice":["VCE"],"Naples":["NAP"],"Bologna":["BLQ"],"Turin":["TRN"]
    }},
    "Netherlands":{"langs":["nl","en"],"cities":{
        "Amsterdam":["AMS"],"Rotterdam":["RTM"],"Eindhoven":["EIN"]
    }},
    "Belgium":{"langs":["nl","fr","de","en"],"cities":{
        "Brussels":["BRU","CRL"],"Antwerp":["ANR"]
    }},
    "Portugal":{"langs":["pt","en"],"cities":{
        "Lisbon":["LIS"],"Porto":["OPO"],"Faro":["FAO"]
    }},
    "Greece":{"langs":["el","en"],"cities":{
        "Athens":["ATH"],"Thessaloniki":["SKG"],"Heraklion":["HER"]
    }},
    "Poland":{"langs":["pl","en"],"cities":{
        "Warsaw":["WAW","WMI"],"Krakow":["KRK"],"Gdansk":["GDN"],"Wroclaw":["WRO"],"Poznan":["POZ"]
    }},
    "Austria":{"langs":["de","en"],"cities":{
        "Vienna":["VIE"],"Salzburg":["SZG"],"Graz":["GRZ"],"Innsbruck":["INN"]
    }},
    "Czechia":{"langs":["cs","en"],"cities":{
        "Prague":["PRG"],"Brno":["BRQ"]
    }},
    "Hungary":{"langs":["hu","en"],"cities":{
        "Budapest":["BUD"],"Debrecen":["DEB"]
    }},
    "Slovakia":{"langs":["sk","en"],"cities":{
        "Bratislava":["BTS"],"Kosice":["KSC"]
    }},
    "Slovenia":{"langs":["sl","en"],"cities":{
        "Ljubljana":["LJU"]
    }},
    "Croatia":{"langs":["hr","en"],"cities":{
        "Zagreb":["ZAG"],"Split":["SPU"],"Dubrovnik":["DBV"]
    }},
    "Romania":{"langs":["ro","en"],"cities":{
        "Bucharest":["OTP","BBU"],"Cluj-Napoca":["CLJ"]
    }},
    "Bulgaria":{"langs":["bg","en"],"cities":{
        "Sofia":["SOF"],"Varna":["VAR"]
    }},
    "Serbia":{"langs":["sr","en"],"cities":{
        "Belgrade":["BEG"],"Nis":["INI"]
    }},
    "Bosnia and Herzegovina":{"langs":["bs","hr","sr","en"],"cities":{
        "Sarajevo":["SJJ"]
    }},
    "Albania":{"langs":["sq","en"],"cities":{
        "Tirana":["TIA"]
    }},
    "North Macedonia":{"langs":["mk","en"],"cities":{
        "Skopje":["SKP"]
    }},
    "Montenegro":{"langs":["sr","en"],"cities":{
        "Podgorica":["TGD"]
    }},
    "Lithuania":{"langs":["lt","en"],"cities":{
        "Vilnius":["VNO"],"Kaunas":["KUN"]
    }},
    "Latvia":{"langs":["lv","en"],"cities":{
        "Riga":["RIX"]
    }},
    "Estonia":{"langs":["et","en"],"cities":{
        "Tallinn":["TLL"]
    }},
    "Finland":{"langs":["fi","sv","en"],"cities":{
        "Helsinki":["HEL"],"Tampere":["TMP"]
    }},
    "Sweden":{"langs":["sv","en"],"cities":{
        "Stockholm":["ARN","BMA"],"Gothenburg":["GOT"],"Malmo":["MMX"]
    }},
    "Norway":{"langs":["no","en"],"cities":{
        "Oslo":["OSL","TRF"],"Bergen":["BGO"],"Stavanger":["SVG"]
    }},
    "Denmark":{"langs":["da","en"],"cities":{
        "Copenhagen":["CPH","RKE"],"Aarhus":["AAR"],"Billund":["BLL"]
    }},

    # AMERICAS
    "USA":{"langs":["en","es"],"cities":{
        "New York":["JFK","EWR","LGA"],"San Francisco":["SFO"],"Los Angeles":["LAX"],"Chicago":["ORD","MDW"],"Miami":["MIA"],"Seattle":["SEA"],"Boston":["BOS"],"Dallas":["DFW","DAL"],"Atlanta":["ATL"],"Washington":["IAD","DCA"]
    }},
    "Canada":{"langs":["en","fr"],"cities":{
        "Toronto":["YYZ","YTZ"],"Vancouver":["YVR"],"Montreal":["YUL","YHU"],"Calgary":["YYC"],"Ottawa":["YOW"]
    }},
    "Mexico":{"langs":["es","en"],"cities":{
        "Mexico City":["MEX","NLU","TLC"],"Guadalajara":["GDL"],"Monterrey":["MTY"],"Cancun":["CUN"]
    }},
    "Brazil":{"langs":["pt","en"],"cities":{
        "Sao Paulo":["GRU","CGH","VCP"],"Rio de Janeiro":["GIG","SDU"],"Brasilia":["BSB"],"Belo Horizonte":["CNF"],"Porto Alegre":["POA"]
    }},
    "Argentina":{"langs":["es","en"],"cities":{
        "Buenos Aires":["EZE","AEP"],"Cordoba":["COR"]
    }},
    "Chile":{"langs":["es","en"],"cities":{
        "Santiago":["SCL"]
    }},
    "Colombia":{"langs":["es","en"],"cities":{
        "Bogota":["BOG"],"Medellin":["MDE"],"Cartagena":["CTG"]
    }},
    "Peru":{"langs":["es","en"],"cities":{
        "Lima":["LIM"],"Cusco":["CUZ"]
    }},

    # MEA / SOUTH ASIA
    "UAE":{"langs":["ar","en"],"cities":{"Dubai":["DXB","DWC"],"Abu Dhabi":["AUH"],"Sharjah":["SHJ"]}},
    "Saudi Arabia":{"langs":["ar","en"],"cities":{"Riyadh":["RUH"],"Jeddah":["JED"],"Dammam":["DMM"]}},
    "Qatar":{"langs":["ar","en"],"cities":{"Doha":["DOH"]}},
    "Jordan":{"langs":["ar","en"],"cities":{"Amman":["AMM"]}},
    "Lebanon":{"langs":["ar","fr","en"],"cities":{"Beirut":["BEY"]}},
    "Israel":{"langs":["he","en"],"cities":{"Tel Aviv":["TLV"],"Eilat":["ETM"]}},
    "Pakistan":{"langs":["ur","en"],"cities":{"Karachi":["KHI"],"Lahore":["LHE"],"Islamabad":["ISB"],"Peshawar":["PEW"],"Quetta":["UET"]}},
    "India":{"langs":["hi","en"],"cities":{"Delhi":["DEL"],"Mumbai":["BOM"],"Bangalore":["BLR"],"Hyderabad":["HYD"],"Chennai":["MAA"],"Kolkata":["CCU"],"Pune":["PNQ"]}},
    "Bangladesh":{"langs":["bn","en"],"cities":{"Dhaka":["DAC"],"Chittagong":["CGP"]}},
    "Sri Lanka":{"langs":["si","ta","en"],"cities":{"Colombo":["CMB"],"Mattala":["HRI"]}},
    "Nepal":{"langs":["ne","en"],"cities":{"Kathmandu":["KTM"]}},
    "Turkey":{"langs":["tr","en"],"cities":{"Istanbul":["IST","SAW"],"Ankara":["ESB"],"Izmir":["ADB"],"Antalya":["AYT"]}},
    "Iran":{"langs":["fa","en"],"cities":{"Tehran":["IKA","THR"],"Shiraz":["SYZ"],"Mashhad":["MHD"]}},
    "Morocco":{"langs":["ar","fr","en"],"cities":{"Marrakech":["RAK"],"Casablanca":["CMN"],"Rabat":["RBA"]}},
    "Tunisia":{"langs":["ar","fr","en"],"cities":{"Tunis":["TUN"]}},
    "Algeria":{"langs":["ar","fr","en"],"cities":{"Algiers":["ALG"]}},
    "Egypt":{"langs":["ar","en"],"cities":{"Cairo":["CAI"],"Alexandria":["HBE"]}},
    "South Africa":{"langs":["en","zu","xh","af"],"cities":{"Johannesburg":["JNB"],"Cape Town":["CPT"],"Durban":["DUR"]}},
    "Kenya":{"langs":["en","sw"],"cities":{"Nairobi":["NBO","WIL"],"Mombasa":["MBA"]}},

    # EAST / SE ASIA + OCEANIA
    "China":{"langs":["zh","en"],"cities":{"Beijing":["PEK","PKX"],"Shanghai":["PVG","SHA"],"Shenzhen":["SZX"],"Guangzhou":["CAN"],"Chengdu":["TFU","CTU"]}},
    "Hong Kong":{"langs":["zh","en"],"cities":{"Hong Kong":["HKG"]}},
    "Taiwan":{"langs":["zh","en"],"cities":{"Taipei":["TPE","TSA"],"Kaohsiung":["KHH"]}},
    "Japan":{"langs":["ja","en"],"cities":{"Tokyo":["HND","NRT"],"Osaka":["KIX","ITM"],"Nagoya":["NGO"],"Fukuoka":["FUK"],"Sapporo":["CTS"]}},
    "South Korea":{"langs":["ko","en"],"cities":{"Seoul":["ICN","GMP"],"Busan":["PUS"]}},
    "Thailand":{"langs":["th","en"],"cities":{"Bangkok":["BKK","DMK"],"Chiang Mai":["CNX"],"Phuket":["HKT"]}},
    "Vietnam":{"langs":["vi","en"],"cities":{"Hanoi":["HAN"],"Ho Chi Minh City":["SGN"],"Da Nang":["DAD"]}},
    "Malaysia":{"langs":["ms","en"],"cities":{"Kuala Lumpur":["KUL","SZB"],"Penang":["PEN"],"Kuching":["KCH"]}},
    "Singapore":{"langs":["en","ms","zh","ta"],"cities":{"Singapore":["SIN"]}},
    "Indonesia":{"langs":["id","en"],"cities":{"Jakarta":["CGK","HLP"],"Bali":["DPS"],"Surabaya":["SUB"]}},
    "Philippines":{"langs":["en","tl"],"cities":{"Manila":["MNL"],"Cebu":["CEB"],"Davao":["DVO"]}},
    "Cambodia":{"langs":["km","en"],"cities":{"Phnom Penh":["PNH"],"Siem Reap":["SAI","REP"]}},
    "Laos":{"langs":["lo","en"],"cities":{"Vientiane":["VTE"],"Luang Prabang":["LPQ"]}},
    "Myanmar":{"langs":["my","en"],"cities":{"Yangon":["RGN"]}},
    "Brunei":{"langs":["ms","en"],"cities":{"Bandar Seri Begawan":["BWN"]}},
    "Australia":{"langs":["en"],"cities":{"Sydney":["SYD"],"Melbourne":["MEL"],"Brisbane":["BNE"],"Perth":["PER"],"Adelaide":["ADL"]}},
    "New Zealand":{"langs":["en"],"cities":{"Auckland":["AKL"],"Wellington":["WLG"],"Christchurch":["CHC"]}},
    "Fiji":{"langs":["en","fj","hi"],"cities":{"Nadi":["NAN"],"Suva":["SUV"]}}
}

CURRENCY_BY_COUNTRY = {
    # Europe
    "UK":"GBP","Germany":"EUR","France":"EUR","Spain":"EUR","Italy":"EUR","Portugal":"EUR","Netherlands":"EUR","Belgium":"EUR",
    "Switzerland":"CHF","Austria":"EUR","Greece":"EUR","Poland":"PLN","Czechia":"CZK","Hungary":"HUF","Slovakia":"EUR","Slovenia":"EUR","Croatia":"EUR",
    "Romania":"RON","Bulgaria":"BGN","Serbia":"RSD","Bosnia and Herzegovina":"BAM","Albania":"ALL","North Macedonia":"MKD","Montenegro":"EUR",
    "Lithuania":"EUR","Latvia":"EUR","Estonia":"EUR","Finland":"EUR","Sweden":"SEK","Norway":"NOK","Denmark":"DKK","Ireland":"EUR",

    # Americas
    "USA":"USD","Canada":"CAD","Mexico":"MXN","Brazil":"BRL","Argentina":"ARS","Chile":"CLP","Colombia":"COP","Peru":"PEN",

    # MEA / South Asia
    "UAE":"AED","Saudi Arabia":"SAR","Qatar":"QAR","Jordan":"JOD","Lebanon":"LBP","Israel":"ILS",
    "Pakistan":"PKR","India":"INR","Bangladesh":"BDT","Sri Lanka":"LKR","Nepal":"NPR","Turkey":"TRY","Iran":"IRR",
    "Morocco":"MAD","Tunisia":"TND","Algeria":"DZD","Egypt":"EGP","South Africa":"ZAR","Kenya":"KES",

    # East / SE Asia + Oceania
    "China":"CNY","Hong Kong":"HKD","Taiwan":"TWD","Japan":"JPY","South Korea":"KRW","Thailand":"THB","Vietnam":"VND","Malaysia":"MYR",
    "Singapore":"SGD","Indonesia":"IDR","Philippines":"PHP","Cambodia":"KHR","Laos":"LAK","Myanmar":"MMK","Brunei":"BND",
    "Australia":"AUD","New Zealand":"NZD","Fiji":"FJD"
}

# Flatten for sampling
LOCATION_DATA = []
for country, c in COUNTRIES.items():
    for city, airports in c["cities"].items():
        LOCATION_DATA.append({"city": city, "country": country, "airports": airports, "languages": c["langs"]})

# =========================================================
# CULTURE-SPECIFIC NAME BANKS (expanded)
# =========================================================
ABDUL_ATTRS = ["Rahman","Rahim","Ghaffar","Qadir","Qahhar","Aziz","Kabir","Jabbar","Rashid","Karim","Hakeem","Basit","Latif","Samad","Salam"]

NAMES = {
    "arabic_gulf": {
        "male_given": ["Omar","Khalid","Yusuf","Faisal","Hamad","Saif","Sultan","Ali","Hassan","Ibrahim"] + [f"Abdul {a}" for a in ABDUL_ATTRS],
        "female_given": ["Aisha","Fatima","Maryam","Noor","Layla","Huda","Reem","Salma","Zainab","Maha"],
        "surnames": ["Al-Harbi","Al-Qahtani","Al-Ansari","Al-Mansoori","Al-Nuaimi","Al-Sabah","Al-Sulimani","Al-Suwaidi","Al-Khalifa","Al-Farsi"]
    },
    "arabic_maghreb": {
        "male_given": ["Youssef","Omar","Khaled","Amine","Rachid","Karim","Hassan"] + [f"Abdul {a}" for a in ABDUL_ATTRS],
        "female_given": ["Amina","Fatima","Khadija","Salma","Nadia","Sara","Imane"],
        "surnames": ["El-Fassi","Bennani","El-Morsi","Hassan","Ramdani","El-Baz","El-Sayed","Bakkali","Tazi","Belkacem"]
    },
    "pakistani": {
        "male_given": ["Ahmed","Imran","Usman","Arslan","Javed","Zeeshan","Hamza","Bilal","Tariq","Noman"] + [f"Abdul {a}" for a in ABDUL_ATTRS],
        "female_given": ["Ayesha","Zainab","Hira","Sana","Iqra","Mariam","Fiza","Mehwish","Laiba","Huma"],
        "surnames": ["Khan","Ahmed","Ali","Hussain","Siddiqui","Malik","Sheikh","Qureshi","Farooq","Ghaffar","Chaudhry","Nawaz"]
    },
    "indian": {
        "male_given": ["Aman","Rahul","Rohit","Arjun","Varun","Vikram","Sarthak","Aditya","Rakesh","Aakash"],
        "female_given": ["Priya","Ananya","Isha","Riya","Aditi","Neha","Kavya","Pooja","Meera","Tanya"],
        "surnames": ["Sharma","Verma","Gupta","Patel","Mehta","Agarwal","Singh","Iyer","Nair","Reddy"]
    },
    "bengali": {
        "male_given": ["Shahid","Arif","Nayeem","Fahim","Shanto","Sazid","Rafi","Siam","Tanvir","Sajib"],
        "female_given": ["Riya","Mim","Nusrat","Jannat","Maliha","Sumaiya","Tanjila","Farzana","Mahira","Afsana"],
        "surnames": ["Chowdhury","Sarkar","Hossain","Rahman","Islam","Karim","Ahmed","Ganguly","Ghosh","Roy"]
    },
    "sinhala_tamil": {
        "male_given": ["Nuwan","Kasun","Shehan","Tharindu","Dilan","Aravind","Pradeep","Vijay","Ramesh","Suren"],
        "female_given": ["Nadeesha","Dilini","Ishara","Anushka","Harini","Aparna","Meera","Kavya","Tharushi","Janani"],
        "surnames": ["Perera","Silva","Fernando","Kumar","Iyer","Nadarajah","Ramanathan","Wijesinghe","Gunasekara","Ratnayake"]
    },
    "nepali": {
        "male_given": ["Sanjay","Prakash","Rabin","Dipesh","Sujan","Anil","Sagar","Raju","Kiran","Roshan"],
        "female_given": ["Sita","Bimala","Bina","Anita","Sujata","Sunita","Sushma","Kanchan","Sabina","Rekha"],
        "surnames": ["Tamang","Rai","Thapa","Gurung","Magar","Khatri","Shrestha","KC","Bhandari","Bhattarai"]
    },
    "persian": {
        "male_given": ["Amir","Reza","Saeed","Hossein","Navid","Arman","Kian","Farid","Pouya","Babak"],
        "female_given": ["Sara","Neda","Leyla","Parisa","Mahnaz","Zahra","Elham","Shadi","Arezoo","Hanieh"],
        "surnames": ["Farhadi","Rahimi","Azizi","Rezai","Hosseini","Ahmadi","Karimi","Sadeghi","Mohammadi","Khodadad"]
    },
    "turkish": {
        "male_given": ["Emre","Mehmet","Ahmet","Can","Mert","Omer","Hakan","Kerem","Yigit","Burak"],
        "female_given": ["Elif","Zeynep","Ayse","Merve","Ece","Nehir","Buse","Seda","Aylin","Selin"],
        "surnames": ["Yilmaz","Demir","Şahin","Çelik","Kaya","Aydin","Yildiz","Öztürk","Arslan","Polat"]
    },
    "maghrebi_fr": {
        "male_given": ["Mehdi","Reda","Yassine","Hicham","Nabil","Karim","Othmane","Sofiane","Walid","Adel"],
        "female_given": ["Ilham","Najat","Samira","Nadia","Houda","Meriem","Rania","Imane","Salwa","Aya"],
        "surnames": ["Amrani","El Idrissi","Belhaj","Fekir","Ouahbi","Bouzid","Zeroual","Ait Said","Chakir","Bouras"]
    },
    "chinese": {
        "male_given": ["Wei","Ming","Jie","Hao","Jun","Tao","Lei","Qiang","Peng","Zhi"],
        "female_given": ["Li","Xiao","Yan","Fang","Mei","Ling","Na","Jing","Ying","Qian"],
        "surnames": ["Wang","Li","Zhang","Liu","Chen","Yang","Huang","Zhao","Wu","Zhou"]
    },
    "japanese": {
        "male_given": ["Haruto","Yuto","Sota","Ren","Daiki","Kaito","Takumi","Ryota","Yuki","Keita"],
        "female_given": ["Yui","Aoi","Sakura","Haruka","Rin","Mio","Hina","Nanami","Ayaka","Mei"],
        "surnames": ["Sato","Suzuki","Takahashi","Tanaka","Watanabe","Ito","Yamamoto","Nakamura","Kobayashi","Kato"]
    },
    "korean": {
        "male_given": ["Min-Jun","Seo-Jun","Ji-Hoon","Hyun-Woo","Joon","Seung-Min","Dong-Hyun","Tae-Hyun","Ji-Ho","Jin"],
        "female_given": ["Seo-Yeon","Ji-Woo","Ha-Young","Min-Seo","Soo-Min","Eun-Ji","Ji-Ah","Hye-Jin","Ye-Jin","Da-Eun"],
        "surnames": ["Kim","Lee","Park","Choi","Jung","Kang","Cho","Yoon","Jang","Lim"]
    },
    "vietnamese": {
        "male_given": ["Anh","Duc","Minh","Huy","Khanh","Nam","Quan","Phuc","Thanh","Trung"],
        "female_given": ["Anh","Trang","Linh","Ngoc","Thao","Phuong","Huong","Quynh","My","Vy"],
        "surnames": ["Nguyen","Tran","Le","Pham","Hoang","Huynh","Phan","Vu","Dang","Bui"]
    },
    "thai": {
        "male_given": ["Thanon","Anan","Somchai","Niran","Kittisak","Suphachai","Arthit","Chai","Prasert","Somsak"],
        "female_given": ["Anong","Kanya","Malee","Suda","Nok","Pim","Dao","Aom","Noi","Mint"],
        "surnames": ["Sukhum","Chantira","Wattanakul","Srisai","Phrom","Sutthikul","Rattanapong","Kittikun","Rungrot","Chaisiri"]
    },
    "malay": {
        "male_given": ["Ahmad","Muhammad","Faiz","Hafiz","Hakim","Amir","Azlan","Irfan","Zul","Imran"],
        "female_given": ["Nur","Aisyah","Farah","Siti","Hani","Amira","Syifa","Balqis","Nadia","Liyana"],
        "surnames": ["bin Abdullah","bin Hassan","bin Ismail","binti Ahmad","binti Ali","bin Rahman","binti Osman","bin Ibrahim","bin Aziz","binti Zain"]
    },
    "indonesian": {
        "male_given": ["Adi","Rizky","Budi","Agus","Fajar","Bayu","Yusuf","Rafli","Dimas","Andi"],
        "female_given": ["Sari","Putri","Ayu","Dewi","Nadia","Nisa","Rani","Tika","Indah","Rina"],
        "surnames": ["Santoso","Saputra","Wijaya","Hidayat","Pratama","Permata","Kurniawan","Ananda","Wibowo","Maulana"]
    },
    "filipino": {
        "male_given": ["Juan","Jose","Paolo","Mark","Jayson","Michael","Christian","Angelo","Carlo","Gabriel"],
        "female_given": ["Maria","Angelica","Anne","Patricia","Andrea","Samantha","Mika","Kimberly","Kathryn","Janelle"],
        "surnames": ["Santos","Reyes","Cruz","Bautista","Garcia","Dela Cruz","Gonzales","Torres","Mendoza","Aquino"]
    },
    "germanic": {
        "male_given": ["Lukas","Jonas","Leon","Maximilian","Felix","Niklas","Tobias","Paul","Moritz","Noah"],
        "female_given": ["Emma","Lea","Mia","Lena","Sophia","Hannah","Laura","Anna","Clara","Luisa"],
        "surnames": ["Müller","Schmidt","Schneider","Fischer","Weber","Meyer","Wagner","Becker","Hoffmann","Schäfer"]
    },
    "french": {
        "male_given": ["Lucas","Hugo","Thomas","Louis","Arthur","Gabriel","Nathan","Theo","Maxime","Antoine"],
        "female_given": ["Emma","Chloé","Camille","Léa","Manon","Zoé","Sarah","Inès","Julie","Lucie"],
        "surnames": ["Martin","Bernard","Dubois","Durand","Lefebvre","Moreau","Laurent","Simon","Michel","Garcia"]
    },
    "italian": {
        "male_given": ["Luca","Matteo","Alessandro","Marco","Leonardo","Simone","Giuseppe","Davide","Gabriele","Antonio"],
        "female_given": ["Giulia","Sofia","Aurora","Martina","Chiara","Alessia","Giorgia","Francesca","Elisa","Federica"],
        "surnames": ["Rossi","Russo","Ferrari","Esposito","Bianchi","Romano","Gallo","Costa","Fontana","Greco"]
    },
    "spanish": {
        "male_given": ["Alejandro","Daniel","Pablo","Javier","Adrián","Sergio","Diego","Mario","Álvaro","Hugo"],
        "female_given": ["Lucía","María","Paula","Sara","Laura","Ana","Carmen","Alba","Claudia","Sofia"],
        "surnames": ["García","Martínez","López","Sánchez","Pérez","González","Rodríguez","Fernández","Díaz","Ruiz"]
    },
    "portuguese": {
        "male_given": ["João","Miguel","Gonçalo","Tiago","Diogo","Rafael","Rodrigo","Pedro","Henrique","Bruno"],
        "female_given": ["Maria","Inês","Ana","Beatriz","Carolina","Margarida","Sara","Mariana","Sofia","Joana"],
        "surnames": ["Silva","Santos","Ferreira","Pereira","Oliveira","Costa","Rodrigues","Martins","Jesus","Sousa"]
    },
    "dutch": {
        "male_given": ["Daan","Sem","Bram","Luuk","Thijs","Lars","Milan","Ties","Finn","Jesse"],
        "female_given": ["Sophie","Julia","Lieke","Anna","Eva","Noa","Zoë","Sara","Luna","Isa"],
        "surnames": ["de Jong","Jansen","de Vries","van den Berg","van Dijk","Bakker","Visser","Smit","Meijer","Bos"]
    },
    "polish": {
        "male_given": ["Jakub","Kacper","Szymon","Antoni","Jan","Filip","Mikolaj","Bartosz","Mateusz","Piotr"],
        "female_given": ["Zuzanna","Julia","Maja","Oliwia","Alicja","Natalia","Wiktoria","Anna","Hanna","Kinga"],
        "surnames": ["Nowak","Kowalski","Wiśniewski","Wójcik","Kamiński","Lewandowski","Zieliński","Szymański","Woźniak","Dąbrowski"]
    },
    "greek": {
        "male_given": ["Giorgos","Dimitris","Nikos","Vasilis","Kostas","Panagiotis","Christos","Giannis","Spyros","Theodoros"],
        "female_given": ["Maria","Eleni","Katerina","Sofia","Dimitra","Ioanna","Georgia","Anna","Despoina","Foteini"],
        "surnames": ["Papadopoulos","Nikolaidis","Georgiou","Papanikolaou","Christodoulou","Karagiannis","Konstantinou","Vasileiou","Ioannou","Economou"]
    },
    "anglo": {
        "male_given": ["Jack","Oliver","Noah","James","William","Henry","Lucas","Ethan","Liam","Samuel"],
        "female_given": ["Olivia","Amelia","Ava","Isla","Mia","Sophia","Emily","Grace","Ella","Charlotte"],
        "surnames": ["Smith","Johnson","Williams","Brown","Jones","Miller","Davis","Wilson","Taylor","Thomas"]
    },
    "latam_spanish": {
        "male_given": ["Juan","Carlos","Luis","Andrés","Fernando","Diego","Jorge","Miguel","Santiago","Ricardo"],
        "female_given": ["María","Andrea","Camila","Valentina","Daniela","Fernanda","Lucía","Paula","Sofía","Carolina"],
        "surnames": ["García","Rodríguez","Martínez","Hernández","González","López","Pérez","Sánchez","Ramírez","Torres"]
    },
    "brazilian": {
        "male_given": ["Lucas","Gabriel","Mateus","Pedro","Gustavo","Rafael","João","Bruno","Thiago","Felipe"],
        "female_given": ["Ana","Maria","Beatriz","Larissa","Camila","Juliana","Fernanda","Luana","Patrícia","Gabriela"],
        "surnames": ["Silva","Santos","Oliveira","Souza","Rodrigues","Ferreira","Almeida","Lima","Gomes","Costa"]
    },
    "slavic_east": {
        "male_given": ["Ivan","Dmitry","Alexei","Sergei","Nikolai","Andrei","Viktor","Pavel","Mikhail","Yuri"],
        "female_given": ["Anna","Olga","Tatiana","Irina","Natalia","Svetlana","Maria","Elena","Daria","Polina"],
        "surnames": ["Ivanov","Petrov","Sidorov","Smirnov","Volkov","Kuznetsov","Morozov","Orlov","Fedorov","Antonov"]
    },
    "nordic": {
        "male_given": ["Lars","Erik","Jon","Anders","Nils","Mats","Sven","Henrik","Karl","Ola"],
        "female_given": ["Ingrid","Anna","Karin","Sara","Linn","Maja","Emilia","Hanna","Freja","Liv"],
        "surnames": ["Andersen","Johansson","Hansen","Larsen","Nielsen","Olsen","Berg","Holm","Lind","Dahl"]
    },
    "balkan": {
        "male_given": ["Marko","Nikola","Stefan","Milan","Luka","Bojan","Dusan","Dragan","Goran","Zoran"],
        "female_given": ["Milica","Jelena","Ana","Marija","Katarina","Ivana","Sara","Teodora","Nevena","Tamara"],
        "surnames": ["Jovanovic","Petrovic","Nikolic","Ilic","Stojanovic","Kovacevic","Savic","Markovic","Popovic","Milosevic"]
    },
    "african_za_ke": {
        "male_given": ["Thabo","Sipho","Kofi","Ayo","Kamau","Juma","Brian","Peter","Daniel","Moses"],
        "female_given": ["Thandi","Zanele","Amina","Nia","Wanjiru","Achieng","Grace","Mary","Esther","Ruth"],
        "surnames": ["Dlamini","Khumalo","Ndlovu","Moyo","Ochieng","Otieno","Mwangi","Muthoni","Naidoo","Pillay"]
    }
}

COUNTRY_CULTURE = {
    # MENA
    "UAE":"arabic_gulf","Saudi Arabia":"arabic_gulf","Qatar":"arabic_gulf","Jordan":"arabic_maghreb","Lebanon":"maghrebi_fr",
    "Egypt":"arabic_maghreb","Morocco":"arabic_maghreb","Tunisia":"arabic_maghreb","Algeria":"arabic_maghreb",
    # South Asia
    "Pakistan":"pakistani","India":"indian","Bangladesh":"bengali","Sri Lanka":"sinhala_tamil","Nepal":"nepali",
    "Iran":"persian","Turkey":"turkish",
    # East Asia
    "China":"chinese","Hong Kong":"chinese","Taiwan":"chinese","Japan":"japanese","South Korea":"korean",
    # SE Asia
    "Vietnam":"vietnamese","Thailand":"thai","Malaysia":"malay","Indonesia":"indonesian","Philippines":"filipino",
    "Singapore":"anglo","Cambodia":"vietnamese","Laos":"thai","Myanmar":"thai","Brunei":"malay",
    # Europe
    "Germany":"germanic","Switzerland":None,"France":"french","UK":"anglo","Ireland":"anglo","Netherlands":"dutch","Belgium":"french",
    "Portugal":"portuguese","Spain":"spanish","Greece":"greek","Poland":"polish","Czechia":"slavic_east","Hungary":"slavic_east",
    "Austria":"germanic","Denmark":"nordic","Norway":"nordic","Sweden":"nordic","Finland":"nordic","Romania":"slavic_east","Bulgaria":"slavic_east",
    "Slovakia":"slavic_east","Slovenia":"balkan","Croatia":"balkan","Serbia":"balkan","Bosnia and Herzegovina":"balkan","Albania":"balkan",
    "North Macedonia":"balkan","Montenegro":"balkan","Lithuania":"slavic_east","Latvia":"slavic_east","Estonia":"slavic_east",
    # Americas
    "USA":"anglo","Canada":"anglo","Mexico":"latam_spanish","Argentina":"latam_spanish","Brazil":"brazilian","Chile":"latam_spanish",
    "Colombia":"latam_spanish","Peru":"latam_spanish",
    # Africa
    "South Africa":"african_za_ke","Kenya":"african_za_ke",
    # Oceania
    "Australia":"anglo","New Zealand":"anglo","Fiji":"anglo"
}

SWISS_LANG_TO_CULT = {"de":"germanic","fr":"french","it":"italian"}

# Basic per-country phone calling code map for realism
COUNTRY_PHONE_CODE = {
    "USA":"+1","Canada":"+1","Mexico":"+52","Brazil":"+55","Argentina":"+54","Chile":"+56","Colombia":"+57","Peru":"+51",
    "UK":"+44","Ireland":"+353","Germany":"+49","France":"+33","Spain":"+34","Italy":"+39","Portugal":"+351","Netherlands":"+31","Belgium":"+32",
    "Switzerland":"+41","Austria":"+43","Greece":"+30","Poland":"+48","Czechia":"+420","Hungary":"+36","Slovakia":"+421","Slovenia":"+386",
    "Croatia":"+385","Romania":"+40","Bulgaria":"+359","Serbia":"+381","Bosnia and Herzegovina":"+387","Albania":"+355","North Macedonia":"+389",
    "Montenegro":"+382","Lithuania":"+370","Latvia":"+371","Estonia":"+372","Finland":"+358","Sweden":"+46","Norway":"+47","Denmark":"+45",
    "UAE":"+971","Saudi Arabia":"+966","Qatar":"+974","Jordan":"+962","Lebanon":"+961","Israel":"+972","Turkey":"+90","Iran":"+98",
    "Pakistan":"+92","India":"+91","Bangladesh":"+880","Sri Lanka":"+94","Nepal":"+977","Egypt":"+20","Morocco":"+212","Tunisia":"+216","Algeria":"+213",
    "South Africa":"+27","Kenya":"+254",
    "China":"+86","Hong Kong":"+852","Taiwan":"+886","Japan":"+81","South Korea":"+82","Thailand":"+66","Vietnam":"+84","Malaysia":"+60",
    "Singapore":"+65","Indonesia":"+62","Philippines":"+63","Cambodia":"+855","Laos":"+856","Myanmar":"+95","Brunei":"+673",
    "Australia":"+61","New Zealand":"+64","Fiji":"+679"
}

# =========================================================
# OTHER POOLS
# =========================================================
GENDERS = ["Male","Female","Non-binary","Other"]
VALUES  = ["adventure","stability","learning","family","budget-minded","luxury-taste","nature","culture","community","fitness","spirituality"]
CAUSES  = ["environmentalism","human rights","disability rights","immigrant rights","LGBTQ+ rights","voter rights","reproductive rights","neurodiversity","end religious hate","stop asian hate","volunteering"]

INTERESTS = [
    "mountain hiking","city photography","food tours","street food","coffee crawls","scenic trains","short hikes","long hikes",
    "nightlife","museum hopping","architecture walks","history sites","skiing","diving","sailing","cycling","festivals",
    "thermal baths","vineyards","wildlife watching","markets","street art","rooftop views","bookstores","local crafts","castles",
    "beach days","lake swims","yoga","trail running","road trips","camping","chess","board games","cinema","live music",
    "theatre","coding","robotics","blogging","vlogging","language exchange","photography","street markets","botanical gardens","football matches"
]
LEARNING_STYLES = ["Visual","Auditory","Kinesthetic","Reading/Writing"]
HUMOR_STYLES    = ["Dry","Witty","Slapstick","Sarcastic","Playful","Observational"]

DIET   = ["none","vegetarian","vegan","halal","kosher","gluten-free","no pork","lactose-free","pescatarian"]
RISK   = ["low","medium","high"]
NOISE  = ["low","medium","high"]
CLEAN  = ["low","medium","high"]
PACE   = ["relaxed","balanced","packed"]
ACCOM_TYPES = ["hotel","apartment","guesthouse","boutique","hostel"]
ROOM_SETUP  = ["twin","double","2 rooms","dorm"]
TRANSPORT   = ["train","plane","bus","car","ferry"]
CHRONO      = ["early bird","night owl","flexible"]
ALCOHOL     = ["none","moderate","social"]
SMOKING     = ["never","occasionally","regular"]

TRIP_INTENTIONS = [
    "weekend city breaks","slow travel & cafes","country-hopping adventure","workation with good wifi",
    "food & culture deep dive","outdoor & hiking focus","beaches & warm weather","photography missions",
    "ski or snow trips","festival/ events chasing","history & museums track","road trips & scenic drives"
]
COMPANION_TYPES = ["solo","pair","small group (3–4)","bigger group (5+)"]
COMPANION_GENDER_PREF = ["any","men","women","nonbinary"]

OPENING_MOVE_TEMPLATES = [
    "Window or aisle—and why?",
    "One hidden gem in your city I shouldn’t miss?",
    "Pick one: sunrise hike or late-night street food crawl?",
    "What’s your undefeated travel snack?",
    "If I plan trains, you plan food—deal?"
]
PROMPTS = {
    "Bit of fun": [
        "Two truths and a travel lie","My favorite way to do nothing is","When my phone dies I",
        "A nickname my travel friends have for me is","I was today years old when I learned"
    ],
    "About me": [
        "I’m known for","My simple pleasures are","After work you can find me","I’m happiest when","The quickest way to my heart is"
    ],
    "Real talk": [
        "A pro and a con of traveling with me","My dream trip is","I get really nervous when",
        "I show I care by","Don’t be mad if I"
    ],
    "Looking for": [
        "What I’d really like to find is","What makes a great travel partner is","My ultimate green flag is",
        "Hopefully you’re also really into","Teach me something about"
    ]
}

# =========================================================
# HELPERS
# =========================================================
def pick_location():
    return random.choice(LOCATION_DATA)

def station_name(city, country):
    if country in ["Germany","Austria","Switzerland"]:
        return f"{city} Hbf"
    if country in ["Poland","Czechia","Hungary","Slovakia","Slovenia","Croatia","Serbia"]:
        return f"{city} Centralna"
    return f"{city} Central Station"

def sample_languages(default_langs):
    base = list(set(default_langs + (["en"] if "en" not in default_langs and random.random() < 0.6 else [])))
    k = random.randint(1, min(3, len(base)))
    return random.sample(base, k=k)

def personality_block():
    return {
        "openness": round(random.uniform(0.15, 0.98), 2),
        "conscientiousness": round(random.uniform(0.15, 0.98), 2),
        "extraversion": round(random.uniform(0.15, 0.98), 2),
        "agreeableness": round(random.uniform(0.15, 0.98), 2),
        "neuroticism": round(random.uniform(0.05, 0.85), 2),
        "creativity": round(random.uniform(0.2, 0.98), 2),
        "empathy": round(random.uniform(0.2, 0.98), 2)
    }

def short_bio(name, city, interests):
    verbs = ["exploring","learning","documenting","sharing","planning","daydreaming about"]
    two = ", ".join(random.sample(interests, k=min(2, len(interests))))
    return f"{name} from {city} loves {two} and enjoys {random.choice(verbs)} memorable trips."

def values_pick():
    return random.sample(VALUES, k=random.choice([2,3]))

def budget_block(country):
    currency = CURRENCY_BY_COUNTRY.get(country, "EUR")
    # adjust rough region budgets
    base_min, base_max = 50, 260
    if country in ["Switzerland","Norway","Denmark","Iceland"] if "Iceland" in CURRENCY_BY_COUNTRY else []:
        base_min, base_max = 90, 320
    if country in ["India","Pakistan","Bangladesh","Vietnam","Indonesia","Philippines"]:
        base_min, base_max = 25, 120
    per_day = random.randint(base_min, base_max)
    return {"type": "per_day", "amount": per_day, "currency": currency, "split_rule": random.choice(["each_own","50/50","custom"])}

def travel_prefs_block():
    # Always try to include "car", cap at 3, preserve order, no duplicates
    picks = ["car"] + random.sample(TRANSPORT, k=random.randint(1, 3))
    seen = set()
    transport_allowed = []
    for x in picks:
        if x not in seen:
            seen.add(x)
            transport_allowed.append(x)
        if len(transport_allowed) == 3:
            break

    return {
        "pace": random.choice(PACE),
        "accommodation_types": random.sample(ACCOM_TYPES, k=random.randint(1,2)),
        "room_setup": random.choice(ROOM_SETUP),
        "transport_allowed": transport_allowed,
        "must_haves": random.sample(
            ["wifi","private_bath","kitchen","workspace","near_station","quiet_room","breakfast"],
            k=random.randint(1,4)
        ),
    }

def diet_health_block():
    return {
        "diet": random.choice(DIET),
        "allergies": random.sample(["none","nuts","shellfish","pollen","gluten","lactose"], k=1),
        "accessibility": random.choice(["none","elevator_needed","reduced_mobility"])
    }

def comfort_block():
    return {
        "risk_tolerance": random.choice(RISK),
        "noise_tolerance": random.choice(NOISE),
        "cleanliness_preference": random.choice(CLEAN),
        "chronotype": random.choice(CHRONO),
        "alcohol": random.choice(ALCOHOL),
        "smoking": random.choice(SMOKING)
    }

def work_block():
    return {
        "remote_work_ok": random.random() < 0.55,
        "hours_online_needed": random.choice([0,1,2]),
        "wifi_quality_needed": random.choice(["normal","good","excellent"])
    }

def random_prompts():
    out = []
    for section, opts in PROMPTS.items():
        if random.random() < 0.75:
            prompt = random.choice(opts)
            answer_bits = [
                "coffee at sunrise","train-window playlists","budget spreadsheets","handy with offline maps",
                "asking locals for food tips","packing light","taking too many photos","always early for trains"
            ]
            ans = random.choice(answer_bits)
            out.append({"section": section, "prompt": prompt, "answer": ans})
    return out[:3]

def opening_move():
    return random.choice(OPENING_MOVE_TEMPLATES)

def companion_preferences():
    age_low = random.randint(20, 26)
    age_high = age_low + random.choice([6,8,10,12,15])
    return {
        "group_preference": random.choice(COMPANION_TYPES),
        "genders_ok": random.sample(COMPANION_GENDER_PREF, k=random.randint(1, len(COMPANION_GENDER_PREF))),
        "age_range_preferred": [age_low, age_high],
        "kids_ok_in_group": random.choice([True, False]),
        "pets_ok_in_group": random.choice([True, False])
    }

def lifestyle_block():
    return {
        "fitness_level": random.choice(["light","moderate","active"]),
        "daily_steps_goal": random.choice(["5-8k","8-12k","12k+"]),
        "caffeine": random.choice(["tea","coffee","either","none"]),
        "sleeping_habits": random.choice(["light sleeper","heavy sleeper","depends"]),
        "snoring_tolerance": random.choice(["fine","prefer not","earplugs ready"])
    }

def boundaries_safety():
    return {
        "quiet_hours": random.choice(["22:00–07:00","23:00–07:00","flexible"]),
        "photo_consent": random.choice(["ask first","ok for private albums","no faces on public socials"]),
        "social_media": random.choice(["share occasionally","no tagging please","fine with tagging"]),
        "substance_boundaries": random.choice(["no drugs","alcohol in moderation","no cigarettes in room"]),
        "share_location_with_group": random.choice([True, False])
    }

def looking_to_meet_block():
    return {"looking_to_meet": random.sample(["men","women","nonbinary","everyone"], k=1)[0]}

def culture_for(country, city_langs):
    if country == "Switzerland":
        lang = next((l for l in city_langs if l in ("de","fr","it")), "de")
        return SWISS_LANG_TO_CULT.get(lang, "germanic")
    return COUNTRY_CULTURE.get(country, "anglo")

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def build_name(country, gender):
    # choose a bank by culture
    loc = next((x for x in LOCATION_DATA if x["country"] == country), None)
    city_langs = loc["languages"] if loc else ["en"]
    cult = culture_for(country, city_langs)
    bank = NAMES.get(cult, NAMES["anglo"])

    if gender.lower().startswith("male"):
        given_pool = bank["male_given"]
    elif gender.lower().startswith("female"):
        given_pool = bank["female_given"]
    else:
        given_pool = random.choice([bank["male_given"], bank["female_given"]])

    given = random.choice(given_pool)
    family = random.choice(bank["surnames"])
    # Order “Family Given” for CJK/VN/TH to keep style recognizable but easy to parse
    if cult in ("chinese","japanese","korean","vietnamese","thai"):
        full = f"{family} {given}"
    else:
        full = f"{given} {family}"
    return full

# --- New helpers for top-level signup fields ---
def split_name(full_name: str):
    parts = full_name.replace("  ", " ").split()
    first = parts[0] if parts else ""
    last = parts[-1] if len(parts) > 1 else ""
    middle = " ".join(parts[1:-1]) if len(parts) > 2 else ""
    return first, middle, last

def fake_phone(country: str):
    cc = COUNTRY_PHONE_CODE.get(country, "+1")
    return f"{cc}-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"

def fake_street():
    return f"{random.randint(12, 9999)} {random.choice(['Maple','Oak','Cedar','Lake','Sunset','Riverside','Hill','Garden','Park','King','Queen','Pine','Elm','Birch'])} {random.choice(['St','Ave','Rd','Blvd','Ln'])}"

def fake_postal(country: str):
    # simple patterns to look realistic without heavy regex per country
    if country in {"USA"}:
        return f"{random.randint(10000, 99999)}"
    if country in {"India","Pakistan","Bangladesh"}:
        return f"{random.randint(100000, 999999)}"
    if country in {"UK"}:
        return random.choice(["SW1A 1AA","EC1A 1BB","B33 8TH","W1A 0AX","M1 1AE"])
    if country in {"Canada"}:
        return random.choice(["M5V 2T6","H3Z 2Y7","V6B 1V2","K1A 0B1"])
    if country in {"Ireland"}:
        return random.choice(["D02 X285","D08 VF8H","T12 X70A"])
    if country in {"Germany","Austria","Switzerland"}:
        return f"{random.randint(1000, 99999)}"
    if country in {"Netherlands"}:
        return f"{random.randint(1000,9999)} {random.choice(['AA','AB','AC','BA','BB','BC'])}"
    if country in {"France"}:
        return f"{random.randint(10000, 95999)}"
    if country in {"Australia"}:
        return f"{random.randint(200, 9944)}"
    # default numeric
    return f"{random.randint(1000, 999999)}"

def dob_from_age(age: int):
    today = date.today()
    year = today.year - age
    day_of_year = random.randint(1, 365)
    try:
        born = date(year, 1, 1) + timedelta(days=day_of_year-1)
    except Exception:
        born = date(year, 6, 15)
    return datetime(born.year, born.month, born.day).isoformat(timespec="milliseconds") + "Z"

def email_from_name(full_name: str):
    base = _strip_accents(full_name).replace("'", "").replace("’", "")
    parts = base.lower().replace("-", " ").split()
    first_token = parts[0] if parts else "user"
    last_token  = parts[-1] if len(parts) > 1 else "rm"
    return f"{first_token}.{last_token}@rovermitra.example"

# =========================================================
# BUILDERS
# =========================================================
def build_user():
    loc = pick_location()
    city, country, airports, langs = loc["city"], loc["country"], loc["airports"], loc["languages"]

    gender = random.choice(GENDERS)
    name   = build_name(country, gender)

    email  = email_from_name(name)
    handle = "rm_" + _strip_accents(name).lower().replace(" ", ".").replace("-", ".")

    age = random.randint(19, 62)
    languages = sample_languages(langs)
    interests = random.sample(INTERESTS, k=random.randint(5, 10))
    
    # Generate unique user_id
    user_id = f"user_{uuid.uuid4().hex[:12]}"

    # --- NEW top-level signup fields (without removing any existing fields) ---
    firstName, middleName, lastName = split_name(name)
    password = "Pass@" + str(random.randint(100000, 999999))
    signup_block = {
        "email": email,
        "password": password,
        "confirmPassword": password,
        "firstName": firstName,
        "lastName": lastName,
        "middleName": middleName,
        "phoneNumber": fake_phone(country),
        "dateOfBirth": dob_from_age(age),
        "address": fake_street(),
        "city": city,
        "state": "",                   # left blank; varies by country
        "postalCode": fake_postal(country),
        "country": country
    }

    profile = {
        # Include user_id for compatibility with other scripts
        "user_id": user_id,

        # 🔹 New required signup fields at top-level
        **signup_block,

        "name": name,
        "age": age,
        "gender": gender,

        "contact": {"rovermitra_handle": handle, "email": email},

        "home_base": {
            "city": city, "country": country,
            "nearby_nodes": [station_name(city, country), random.choice(airports)],
            "willing_radius_km": random.choice([25, 40, 60, 80])
        },

        "languages": languages,
        "interests": interests,
        "values": values_pick(),
        "personality": personality_block(),
        "bio": short_bio(name, city, interests),

        "budget": budget_block(country),
        "diet_health": diet_health_block(),
        "comfort": comfort_block(),

        "social": {
            "group_size_ok": random.choice([[1,2],[1,2,3],[2,3,4],[1,2,3,4,5]]),
            "learning_style": random.choice(LEARNING_STYLES),
            "humor_style": random.choice(HUMOR_STYLES),
            "dealbreakers_social": random.sample(
                ["no party hostels","no smoking room","no red-eye travel","no >2 transfers","no dorm rooms"],
                k=random.randint(0,2)
            )
        },

        "work": work_block(),
        "travel_prefs": travel_prefs_block(),

        # Engaging onboarding
        "trip_intentions": random.sample(TRIP_INTENTIONS, k=random.randint(1,3)),
        "companion_preferences": companion_preferences(),
        "lifestyle": lifestyle_block(),
        "boundaries_safety": boundaries_safety(),
        "causes": random.sample(CAUSES, k=random.randint(0,3)),
        "prompts": random_prompts(),
        "opening_move": opening_move(),
        **looking_to_meet_block(),

        "privacy": {
            "share_profile_with_matches": True,
            "share_itinerary_with_group": random.random() < 0.9,
            "marketing_opt_in": random.random() < 0.3
        }
    }
    return profile

def degrade_profile_randomly(profile):
    """Make some profiles intentionally less complete (keep core; do NOT remove new signup fields)."""
    removable = [
        "values","personality","bio","diet_health","comfort","social","work","travel_prefs",
        "trip_intentions","companion_preferences","lifestyle","boundaries_safety","causes","prompts","opening_move"
    ]
    k = random.randint(3, 6)
    for key in random.sample(removable, k=min(k, len(removable))):
        profile.pop(key, None)
    return profile

# =========================================================
# GENERATE
# =========================================================
def main():
    users = []
    for _ in range(N_USERS):
        p = build_user()
        if random.random() < LEAN_RATIO:
            p = degrade_profile_randomly(p)
        users.append(p)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(users)} users → {OUT_PATH}")
    print(json.dumps(users[0], indent=2, ensure_ascii=False)[:1200] + "\n...")

if __name__ == "__main__":
    main()
