# This transforms Uchoa et al. 2017 XML datasets to the CVRPTW format used by prof. Tilk.

# Imports
from lxml import etree, objectify
from random import randint

# Format relevant data in a class
class Customer():
    def __init__(self, id, xcoord, ycoord, demand, earliest_arrival, latest_departure, service_time):
        self.id = id
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.demand = demand
        self.earliest_arrival = earliest_arrival
        self.latest_departure = latest_departure
        self.service_time = service_time


class SolverDataInput():
    def __init__(self, instance_name):
        self.instance_name = instance_name
        self.vehicles      = 0
        self.capacity      = 0
        self.customers     = []

    def set_vehicles(self, n):
        self.vehicles = n
    
    def set_capacity(self, cap):
        self.capacity = cap
    
    def add_customer(self, id, xcoord, ycoord, demand, earliest_arrival, latest_departure, service_time):
        self.customers.append(Customer(id, xcoord, ycoord, demand, earliest_arrival, latest_departure, service_time))


# Name directory and filename
directory = 'datasets'
output_dir = 'outputs'
filename = 'X-n586-k159'
time_windows = [15, 30, 45, 60]
# Open file
with open('{}/{}.xml'.format(directory,filename),'r') as xml_file:
    xml_doc_readlines = xml_file.readlines()
    xml_file.close()

xml_doc = ''
for line in xml_doc_readlines[1:]:
    xml_doc += line
root = objectify.fromstring(xml_doc)
# Gather data
instance_name = root['info']['name']
intermediate_data = SolverDataInput(instance_name)
fleet_data = [i for i in root.fleet.iterchildren()]
vehicles = len(fleet_data)
capacity = fleet_data[0].capacity
intermediate_data.set_vehicles(vehicles)
intermediate_data.set_capacity(capacity)
nodes = [i for i in root.network.nodes.iterchildren()]
customers = [i for i in root.requests.iterchildren()]
n_customers = len(nodes)
for i in range(n_customers):
    c_id = i
    c_xcoord = nodes[i].cx
    c_ycoord = nodes[i].cy
    c_demand = (customers[i-1].quantity if i > 0 else 0)
    c_sTime = time_windows[randint(0,len(time_windows)-1)]
    c_eArr = (0 if i == 0 else randint(0,859-c_sTime))
    c_lDep = (960 if i == 0 else randint(c_eArr+c_sTime, 860))
    intermediate_data.add_customer(c_id, c_xcoord, c_ycoord, c_demand, c_eArr, c_lDep, c_sTime)
# Format file
    # spacing: 2
    # vehicle slots: 4
    # Capacity: 11
    # Customer slots: 3
    # XCoord slots: 6
    # Remainder: 9
text = intermediate_data.instance_name+'\n\nVEHICLE\nNUMBER     CAPACITY\n'
if len(str(intermediate_data.vehicles)) > 4:
    print("Vehicle length NOT ALLOWED")
    exit()
else:
    v_format = "    "
    v_format = v_format[0:4-len(str(intermediate_data.vehicles))] + str(intermediate_data.vehicles)
if len(str(intermediate_data.capacity)) > 11:
    print("vehicle capacity NOT ALLOWED")
    exit()
else:
    c_format = "           "
    c_format = c_format[0:11-len(str(intermediate_data.capacity))] + str(intermediate_data.capacity)
text += v_format + "  " + c_format + "\n\nCUSTOMER\nCUST NO.   XCOORD.   YCOORD.   DEMAND    READY TIME   DUE DATE   SERVICE TIME\n\n"
for cust in intermediate_data.customers:
    cust_id_format = "   "
    if len(str(cust.id)) > 3:
        print("customers amount NOT ALLOWED")
        exit()
    else:
        cust_id_format = cust_id_format[0:3-len(str(cust.id))] + str(cust.id)
    cust_x_format = "      "
    if len(str(cust.xcoord)) > 6:
        print("xdist NOT ALLOWED")
        exit()
    else:
        cust_x_format = cust_x_format[0:6-len(str(cust.xcoord))] + str(cust.xcoord)
    cust_y_format = "         "
    if len(str(cust.ycoord)) > 9:
        print("ydist NOT ALLOWED")
        exit()
    else:
        cust_y_format = cust_y_format[0:9-len(str(cust.ycoord))] + str(cust.ycoord)
    cust_d_format = "         "
    if len(str(cust.demand)) > 9:
        print("demand NOT ALLOWED")
        exit()
    else:
        cust_d_format = cust_d_format[0:9-len(str(cust.demand))] + str(cust.demand)
    cust_arr_format = "         "
    if len(str(cust.earliest_arrival)) > 9:
        print("earliest_arrival NOT ALLOWED")
        exit()
    else:
        cust_arr_format = cust_arr_format[0:9-len(str(cust.earliest_arrival))] + str(cust.earliest_arrival)
    cust_dep_format = "         "
    if len(str(cust.latest_departure)) > 9:
        print("latest_departure NOT ALLOWED")
        exit()
    else:
        cust_dep_format = cust_dep_format[0:9-len(str(cust.latest_departure))] + str(cust.latest_departure)
    cust_st_format = "         "
    if len(str(cust.service_time)) > 9:
        print("service_time NOT ALLOWED")
        exit()
    else:
        cust_st_format = cust_st_format[0:9-len(str(cust.service_time))] + str(cust.service_time)
    text += "  "+cust_id_format+"  "+cust_x_format+"  "+cust_y_format+"  "+cust_d_format+"  "+cust_arr_format+"  "+cust_dep_format+"  "+cust_st_format+"\n"
with open("{}/{}.txt".format(output_dir,filename), "w") as text_file:
    text_file.write(text)
    text_file.close()