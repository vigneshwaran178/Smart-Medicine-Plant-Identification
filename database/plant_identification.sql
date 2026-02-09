-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Feb 18, 2023 at 10:08 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `plant_identification`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `plant`
--

CREATE TABLE `plant` (
  `id` int(11) NOT NULL,
  `plant` varchar(30) NOT NULL,
  `imgname` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `plant`
--

INSERT INTO `plant` (`id`, `plant`, `imgname`) VALUES
(1, 'Omnavalli', 'image1.jpg'),
(2, 'Omnavalli', 'image2.jpg'),
(3, 'Omnavalli', 'image3.jpg'),
(4, 'Pudina', 'image4.jpg'),
(5, 'Vethalai', 'image5.jpg');

-- --------------------------------------------------------

--
-- Table structure for table `plant_uses`
--

CREATE TABLE `plant_uses` (
  `id` int(11) NOT NULL,
  `plant` varchar(100) NOT NULL,
  `uses` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `plant_uses`
--

INSERT INTO `plant_uses` (`id`, `plant`, `uses`) VALUES
(1, 'Alpinia Galanga (Rasna)', 'Alpinia officinarum Hance (galanga) is a perennial ginger family plant. Galanga has been traditionally used for many years to treat several different diseases including cold, pain, inflammation, stomach ache, and microbial infection, and it also works as an antioxidant and anticancer agent.'),
(2, 'Alpinia Galanga (Rasna)', 'May fight inflammation and pain.'),
(3, 'Alpinia Galanga (Rasna)', 'May protect against infections.'),
(4, 'Amaranthus Viridis (Arive-Dantu)', 'There are plenty of ways to enjoy amaranth as a part of your daily diet: Boil whole amaranth grain in a 3/1 ratio of water to amaranth to make porridge.'),
(5, 'Artocarpus Heterophyllus (Jackfruit)', 'Jackfruit (Artocarpus heterophyllus Lam) is a rich source of several high-value compounds with potential beneficial physiological activities'),
(6, 'Artocarpus Heterophyllus (Jackfruit)', 'Jackfruit offers several key nutrients, including fiber, protein, vitamin C, B vitamins, and potassium.'),
(7, 'Azadirachta Indica (Neem)', 'In the Indian subcontinent, neem leaves are used to treat dental and gastrointestinal disorders, malaria fevers, skin diseases, and as insects repellent, while the Balinese used neem leaves as a diuretic and for diabetes, headache, heartburn, and stimulating the appetite.'),
(8, 'Basella Alba (Basale)', 'Basella alba is reported to improve testosterone levels in males, thus boosting libido. Decoction of the leaves is recommended as a safe laxative in pregnant women and children. Externally, the mucilaginous leaf is crushed and applied in urticaria, burns and scalds.'),
(9, 'Brassica Juncea (Indian Mustard)', 'It is a folk remedy for arthritis, foot ache, lumbago and rheumatism. Brassica juncea is grown mainly for its seed used in the fabrication of brown mustard or for the extraction of vegetable oil. Brown mustard oil is used against skin eruptions and ulcers.'),
(10, 'Carissa Carandas (Karanda)', 'Its fruit is used in the ancient Indian herbal system of medicine, Ayurvedic, to treat acidity, indigestion, fresh and infected wounds, skin diseases, urinary disorders and diabetic ulcer, as well as biliousness, stomach pain, constipation, anemia, skin conditions, anorexia and insanity.'),
(11, 'Citrus Limon (Lemon)', 'limon essential oil was administered on sugar for suppressing coughs [3]. Aside from being rich in vitamin C, which assists in warding off infections, the juice is traditionally used to treat scurvy, sore throats, fevers, rheumatism, high blood pressure, and chest pain'),
(12, 'Ficus Auriculata (Roxburgh fig)', 'Leaves are lopped for fodder. Stem bark juice is effective for diarrhea, cuts and wounds. Fruits are edible and can be made into jams and curries. Roasted figs are taken for diarrhea and dysentery. Root latex is used in mumps, cholera, diarrhea and vomiting.'),
(13, 'Ficus Religiosa (Peepal Tree)', 'Ficus religiosa (L.), commonly known as pepal belonging to the family Moraceae, is used traditionally as antiulcer, antibacterial, antidiabetic, in the treatment of gonorrhea and skin diseases.'),
(14, 'Hibiscus Rosa-sinensis', 'The Hibiscus rosa-sinensis flower is widely used in Brazilian traditional medicine for the treatment of diabetes and has shown antifertility activity in female Wistar rats. However, there is no scientific confirmation of its effect on diabetes and pregnancy.'),
(15, 'Jasminum (Jasmine)', 'Jasmine is inhaled to improve mood, reduce stress, and reduce food cravings. In foods, jasmine is used to flavor beverages, frozen dairy desserts, candy, baked goods, gelatins, and puddings. In manufacturing, jasmine is used to add fragrance to creams, lotions, and perfumes.'),
(16, 'Mangifera Indica (Mango)', 'Various parts of plant are used as a dentrifrice, antiseptic, astringent, diaphoretic, stomachic, vermifuge, tonic, laxative and diuretic and to treat diarrhea, dysentery, anaemia, asthma, bronchitis, cough, hypertension, insomnia, rheumatism, toothache, leucorrhoea, haemorrhage and piles.'),
(17, 'Mentha (Mint)', 'Among medicinal plants, mint (Mentha species) exhibits multiple health beneficial properties, such as prevention from cancer development and anti-obesity, antimicrobial, anti-inflammatory, anti-diabetic, and cardioprotective effects, as a result of its antioxidant potential, combined with low toxicity and high efficacy'),
(18, 'Moringa Oleifera (Drumstick)', 'Almost all parts of the plant: root, bark, gum, leaf, fruit (pods), flowers, seeds and seed oil, have been used to treat various diseases, like skin infections, swelling, anaemia, asthma, bronchitis, diarrhoea, headache, joint pain, rheumatism, gout, diarrhoea, heart problems, fevers, digestive disorders, wounds'),
(19, 'Muntingia Calabura (Jamaica Cherry-Gasagase)', 'The flowers are used as an antiseptic and to treat abdominal cramps and spasms. It is also taken to relieve headaches and colds. Muntingia calabura fruits possess antioxidant property. However, their anti-inflammatory activity has not been investigated so far.'),
(20, 'Murraya Koenigii (Curry)', 'The green leaves of M. koenigii are used in treating piles, inflammation, itching, fresh cuts, dysentery, bruises, and edema. The roots are purgative to some extent. They are stimulating and used for common body aches.'),
(21, 'Nerium Oleander (Oleander)', 'Its ethnomedicinal uses include treatment of diverse ailments such as heart failure, asthma, corns, cancer, diabetes, and epilepsy. Less well appreciated are the skin care benefits of extracts of N. oleander that include antibacterial, antiviral, immune, and even antitumor properties associated with topical use.'),
(22, 'Nyctanthes Arbor-tristis (Parijata)', 'Ethnopharmacological relevance: Nyctanthes arbor-tristis (Oleaceae) is a mythological plant; has high medicinal values in Ayurveda. The popular medicinal use of this plant are anti-helminthic and anti-pyretic besides its use as a laxative, in rheumatism, skin ailments and as a sedative.'),
(23, 'Ocimum Tenuiflorum (Tulsi)', 'This plant is well known for its medicinal and spiritual properties in Ayurveda which includes aiding cough, asthma, diarrhea, fever, dysentery, arthritis, eye diseases, indigestion, gastric ailments.'),
(24, 'Piper Betle (Betel)', 'Since antiquity, Piper betel. Linn, commonly known as betel vine, has been used as a religious, recreational and medicinal plant in Southeast Asia. The leaves, which are the most commonly used plant part, are pungent with aromatic flavor and are widely consumed as a mouth freshener.'),
(25, 'Plectranthus Amboinicus (Mexican Mint)', 'It is widely used in folk medicine to treat conditions like cold, asthma, constipation, headache, cough, fever and skin diseases. The leaves of the plant are often eaten raw or used as flavoring agents, or incorporated as ingredients in the preparation of traditional food.'),
(26, 'Pongamia Pinnata (Indian Beech)', 'Pongamia pinnata has been applied as crude drug for the treatment of tumors, piles, skin diseases, and ulcers. The root is effective for treating gonorrhea, cleaning gums, teeth, and ulcers, and is used in vaginal and skin diseases.'),
(27, 'Psidium Guajava (Guava)', 'Guava, Psidium guajava (Linn.), a member of Myrtaceae family, is a common tropical plant with a long history of traditional usage. It is used not only as food but also as folk medicine, and various parts of this plant have a number of medicinal properties ranging from antimicrobial activity to anticancer property.'),
(28, 'Punica Granatum (Pomegranate)', 'Accumulating data clearly claimed that Punica granatum L. (pomegranate) has several health benefits. Pomegranates can help prevent or treat various disease risk factors including high blood pressure, high cholesterol, oxidative stress, hyperglycemia, and inflammatory activities.'),
(29, 'Santalum Album (Sandalwood)', 'Sandalwood has antipyretic, antiseptic, antiscabetic, and diuretic properties. It is also effective in treatment of bronchitis, cystitis, dysuria, and diseases of the urinary tract.'),
(30, 'Syzygium Cumini (Jamun)', 'The bark is acrid, sweet, digestive, astringent to the bowels, anthelmintic and used for the treatment of sore throat, bronchitis, asthma, thirst, biliousness, dysentery and ulcers. It is also a good blood purifier.'),
(31, 'Syzygium Jambos (Rose Apple)', 'Species of this genus, including S. jambos, offer edible fruits found under various formulation including juices, jellies, and jams (Sun et al., 2020). The decoction of these fruits serves to alleviate gastrointestinal disorders, wounds, syphilis, leprosy, as well as toothache'),
(32, 'Tabernaemontana Divaricata (Crape Jasmine)', 'Tabernaemontana divaricata (TD) from Apocynaceae family offers the traditional folklore medicinal benefits such as an anti-epileptic, anti-mania, brain tonic, and anti-oxidant.'),
(33, 'Trigonella Foenum-graecum (Fenugreek)', 'Besides its known medicinal properties such as carminative, gastric stimulant, antidiabetic and galactogogue (lactation-inducer) effects, newer research has identified hypocholesterolemic, antilipidemia, antioxidant, hepatoprotective, anti-inflammatory, antibacterial, antifungal, antiulcer, antilithigenic.');

-- --------------------------------------------------------

--
-- Table structure for table `register`
--

CREATE TABLE `register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `email` varchar(30) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `register`
--

INSERT INTO `register` (`id`, `name`, `email`, `mobile`, `uname`, `pass`) VALUES
(1, 'Dinesh', 'dinesh@gmail.com', 9054621096, 'dinesh', '1234');
