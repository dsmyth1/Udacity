import xml.etree.cElementTree as ET
import re

OSMFILE = "interpreter.osm"
OUTPUT = "interpreter_OUTPUT.osm"
problem_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
phone_format_re = re.compile(r'(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$', re.IGNORECASE)

mapping = { "african_methodist_episcopal": "African_Methodist_Episcopal",
           "african_methodist_episcopal_zion": "African_Methodist_Episcopal_Zion", "ame_zion": "Ame_Zion",
           "baptist": "Baptist", "buddhist": "Buddhist", "catholic": "Catholic", "christian": "Christian",
           "episcopal": "Episcopal", "greek_orthodox": "Greek_Orthodox", "jehovahs_witness": "Jehovahs_Witness",
           "jewish": "Jewish", "lutheran": "Lutheran", "methodist": "Methodist", "mormon": "Mormon",
           "muslim": "Muslim", "orthodox": "Orthodox", "pentecostal": "Pentecostal", "presbyterian": "Presbyterian",
           "roman_catholic": "Roman_Catholic", "unitarian_universalist": "Unitarian_Universalist",
           
           "Baltimore (city),Maryland,Md.,MD,USA": "Baltimore, Md.", "neighbourhood": "neighborhood",
           
           "american": "American", "american;burger;pizza;grill": "American;burger;pizza;grill", "arab": "Arab",
           "argentinian": "Argentinian", "asian": "Asian", "chinese": "Chinese", "ethiopian": "Ethiopian",
           "ethopian": "Ethiopian", "french": "French", "greek;regional": "Greek;regional", "hispanic": "Hispanic",
           "indian": "Indian", "indian;nepali": "Indian;Nepali", "italian": "Italian", "italian,pizza": "Italian,pizza",
           "jamaican": "Jamaican", "korean": "Korean", "korean;japanese;chinese": "Korean;Japanese;Chinese",
           "latin": "Latin", "lebanese": "Lebanese", "mediterranean": "Mediterranean", "mexican": "Mexican",
           "middle_eastern": "Middle_Eastern", "peruvian": "Peruvian", "spanish": "Spanish", "thai": "Thai",
           "turkish": "Turkish", "vietnamese": "Vietnamese",
           }
            
def audit_dm(osmfile, output):
    osm_file = open(osmfile, "r")
    output =  open(output, 'wb')
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write('<osm>\n  ')
    
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_problem(tag):
                    tag.attrib['v'] = fix_problem(tag.attrib['v'], mapping)
                    #print tag.attrib['v']  
                elif is_phone(tag):
                    tag.attrib['v'] = format_phone(tag.attrib['v'])
                    #print tag.attrib['v'] 
        output.write(ET.tostring(elem, encoding='utf-8'))

    output.write('</osm>')
            
    osm_file.close()
    output.close()

def is_problem(elem):
    return (elem.attrib['k'] == "religion" or elem.attrib['k'] == "denomination" or  elem.attrib['k'] == "place" or elem.attrib['k'] == "is_in"
            or elem.attrib['k'] == "cuisine")
    
def is_phone(elem):
    return (elem.attrib['k'] == "phone")

def format_phone(phonenumber):                                                                                                                                 
    # remove non-digit characters
    phonenumber = ''.join(ele for ele in phonenumber if ele.isdigit())
    
    #remove leading 1-
    if phonenumber.startswith('1'):
            phonenumber = phonenumber[1:]
    elif phonenumber.startswith('01'):
        phonenumber = phonenumber[:02]
        
    # One erroneous # of 11 digit length, taking first 10
    if len(phonenumber) == 11:
        phonenumber = format_phone(phonenumber[:10])
        
    #Recursion!!! To deal with two listed phone #'s
    if len(phonenumber) == 10:
        phonenumber = ('{0}-{1}-{2}'.format(phonenumber[:3], phonenumber[3:6], phonenumber[6:]))
    elif len(phonenumber) == 21:
        phonenumber = format_phone(phonenumber[:10]) + " " + format_phone(phonenumber[11:21])
        
    return phonenumber

def fix_problem(problem, mapping):
    
    problem_type_re = re.compile(r'\D+\.?$', re.IGNORECASE)
    # Changed from: problem_type_re = re.compile(r'\S+\.?$', re.IGNORECASE)
    #to allow white space in Baltimore (city),Maryland,Md.,MD,USA
    match = problem_type_re.search(problem)
    if match:
        problem_type = match.group()
        if problem_type in mapping.keys():
            mapping.get(problem_type)
            problem = problem.replace(problem_type, mapping.get(problem_type))
    return problem

def test():
    
    audit_dm(OSMFILE, OUTPUT)
            
if __name__ == '__main__':
    test()