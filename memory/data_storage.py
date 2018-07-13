
class Snack:

    def __init__(self, name, price):
        self.name = name
        self.price = price

    def get_price(self):
        return self.price

    def get_name(self):
        return self.name


snacks = [Snack("chocolate", 1), Snack("chips", 0.75), Snack("drink", 0.5)]


class Employee:
    total = 0
    def __init__(self, name):
        self.name = name

    def add_to_total(self, amount):
        self.total = self.total + amount

    def get_total(self):
        return self.total

    def get_name(self):
        return self.name

employees = ["Ashraf Abdelraouf Fathy Abdelshakour",
"Wassim Abou-Melhem",
"Ramesh Adusumilli",
"Peter Joseph Aernoudts",
"Lana Al-Salem Saqer",
"Vasudeva Rao Aourpally",
"Andre Jorge Asano",
"Denis Balazuc",
"Cristhian Balderas",
"Eliseu Folego Baldo",
"Amanda Baldwin",
"Arran Gregory Bartish",
"Ariella Baston",
"Yacine Belala",
"Tatiana Bogatchkina",
"Sebastien Boissonneault",
"Ioana Nora Burdujoc",
"Richard Bussiere",
"Antonio Francesco Caligiuri",
"Nicole Gina Calinoiu",
"Juan Enrique Cardenas-Medina",
"Cedric Caron",
"David Caya",
"Eyad Chikh-ibrahim",
"Manish Chugh",
"Richard Comtois",
"Flavius Costa",
"Sean Coull",
"Valerie Demeule",
"Gabriel Desjardins",
"Angelo Dipaolo",
"Dang Khoa Do",
"Karine Dupont",
"Luis Alberto Estefan Elola",
"Serra Erkal",
"Georges Fadous",
"Laurent Fernandez",
"Diana Firdjanova",
"Samantha Fornari",
"John Gammon",
"Modestos Glykis-Vergados",
"Matthew Benchimol",
"Anastasiia Drozdova",
"Andres Nunez Zegarra",
"Assen Garbev",
"Martin Godin",
"Kasivisweswara Sharma Gollapudi",
"Florian Gombert",
"Ke Gong",
"Atheer Hanna",
"Maddison Harder",
"Gabriel Hernandez",
"Erin Hollen",
"Tenveer Hussain",
"Raynold JR Jean",
"David W Johnson",
"Benoit Joly",
"Thomas Joubin",
"Behnam Karimi",
"Bruce M Kearney",
"Mohamed Kleit",
"Vyacheslav Kostin",
"Sergey Kovalenko",
"Stephanie Kulovics",
"Arvind Kumar",
"Eric Langlais",
"Jean-Philippe LeBlanc",
"Lindy Li Loong",
"Julie Anne Lichocki",
"Quan Liu",
"Anne Lizotte",
"Amarbir Singh Lubana",
"Cherifa Mansoura Liamani",
"Paul Anthony Moore",
"Olivier Mornet",
"Deyvisson Oliveira",
"Erwin Pant",
"Jaswanth Paruchuri",
"Liza Perreault",
"Oksana  Poliarush",
"Leonardo Postacchini",
"Aneesh Potluri",
"Maria-Luisa Quintana",
"Seddik Ramdani",
"Muhammad Ammar",
"Bruno Luiz S Ribeiro",
"Nicolas Richard",
"Sergio Romero Del Bosque",
"Sergei Rybakov",
"Venkata R Sathi",
"Ralf Schneider",
"Sasitharan Sellathurai",
"Vithiyatharan Sellathurai",
"Yury Shamne",
"Abhishek Sharma",
"Nidhi Sharma",
"Dzvenyslava Siatetska",
"Roxanne Sirois",
"Daniel Tardif",
"Vincent-Julien Tortajada",
"Erik Trepanier",
"Pawel Adam Urban",
"Vannadine Ven",
"Artem Volynets",
"Chirag Vyas",
"Vaishalibahen Vyas",
"Ruo Nong Wang",
"Zhihua Xi",
"Jie Xu",
"Larry Belanger",
"Ron Andrew Callender",
"Leah Cerro",
"Jason Cote",
"Beverly Darlene Dentry",
"Feigang Fei",
"Patrick Guindon-Slater",
"Maria Eleftheria Hatajlo",
"Jean-Francois Lemire",
"Steven McGurn",
"Jeannie Riel",
"Jean Nicholas Thomas",
"Oleg Volkovich"
]

def get_employee_names_list():
    return employees