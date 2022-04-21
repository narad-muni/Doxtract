import random
# import faker
from faker import Faker
faker = Faker()

data = """Invoice number
Invoice date
PO number
Date
Due date
Sub total
Total
GST number
Account number
PAN number
Bank name
bank
dt
project name
project
project number
GSTIN
State
city
tax
invoice date
VAT
Invoice#
invoice #
Grand total
Discount total
Sales tax
SGST
CGST
company from
company to
bill from
bill to
invoice to
Dispatch to
dispatch from
dispatch
dispatch date
order number
order id
name
type
supplier
voucher number
dispatch through
quantity
total weigth
total value
phone number
phone
contact number
contact no
customer
customer care number
email id
email
country
bill to
bill from
ship to
ship from
from
to
company""".split("\n")

def panNumber():
    pan = "A"
    status = "T F H P C A".split(" ")
    first = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ")
    for i in range(2):
        pan += random.choice(first)
    pan += random.choice(status)
    pan += random.choice(first)
    for i in range(4):
        pan += str(random.randint(0, 9))
    pan += random.choice(first)
    return pan

def gstNumber():
    gst = "2"
    gst += str(random.randint(1, 9))
    gst += panNumber()
    gst += str(random.randint(1, 9))
    gst += "Z"
    gst += str(random.randint(1, 9))
    return gst

dataType = """number
date
number
date
date
number
number
GST
number
PAN
company
company
date
name
number
number
GST
city
city
number
date
number
number
number
number
number
number
number
number
company
company
company
company
name
name
name
name
date
number
number
name
number
company
number
name
number
number
number
phone
phone
phone
phone
name
phone
email
email
country
companyInfo
companyInfo
companyInfo
companyInfo
companyInfo
companyInfo
companyInfo""".split("\n")

f = open('data.txt',"w+")

for j in range(100):
    shuffled = list(zip(data,dataType))
    random.shuffle(shuffled)
    data,dataType = zip(*shuffled)
    for i in range(len(data)):
        fakeData = ""
        if(dataType[i]=="date"):
            fakeData = faker.date()
        elif(dataType[i]=="number"):
            fakeData = str(random.randint(0,999999))
        elif(dataType[i]=="company"):
            fakeData = faker.company()
        elif(dataType[i]=="name"):
            fakeData = faker.name()
        elif(dataType[i]=="GST"):
            fakeData = gstNumber()
        elif(dataType[i]=="PAN"):
            fakeData = panNumber()
        elif(dataType[i]=="phone"):
            fakeData = faker.phone_number()
        elif(dataType[i]=="email"):
            fakeData = faker.ascii_email()
        elif(dataType[i]=="city"):
            fakeData = faker.city()
        elif(dataType[i]=="country"):
            fakeData = faker.country()
        elif(dataType[i]=="companyInfo"):
            fakeData = faker.company()
            fakeData += "----";
            fakeData += faker.address()
        fakeData = fakeData.replace("\n","(NewLine)")
        f.write(data[i]+"@@@@"+fakeData+"\n")
