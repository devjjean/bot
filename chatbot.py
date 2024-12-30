import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode  # Para remover acentos
import matplotlib.pyplot as plt
import networkx as nx

# Baixar dados do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializando lematizador
lemmatizer = WordNetLemmatizer()

# Dicionário de perguntas e respostas
perguntas_respostas = {
    "oi": "Olá! Como posso te ajudar?",
    "olá": "Oi! Como posso te ajudar?",
    "qual é a sua função?": "Minha função é ajudar você com suas dúvidas!",
    "quem é você?": "Eu sou um chatbot criado para te ajudar.",
    "qual é o seu objetivo?": "Meu objetivo é te ajudar com informações e responder suas perguntas.",
    "tchau": "Até logo! Foi bom falar com você.",
    "bom dia": "Bom dia! Como posso te ajudar?",
    "boa tarde": "Boa tarde! Como posso te ajudar?",
    "boa noite": "Boa noite! Como posso te ajudar?",
    "qual é o seu nome": "Eu sou um chatbot, não tenho nome, mas você pode me chamar como preferir!",
    "como você está": "Eu sou um chatbot, então não tenho sentimentos, mas estou aqui para te ajudar!",
    "o que você faz": "Eu sou um chatbot desenvolvido para responder suas perguntas e conversar com você!",
    "quanto você pesa": "Eu não tenho corpo, então não peso nada!",
    "qual é a sua idade": "Eu não tenho idade, sou apenas um programa de computador!",
    "você gosta de programar": "Eu sou feito para programar, então sim, eu 'gosto' de programar!",
    "quem te criou": "Eu fui criado por um programador para ajudar a responder suas perguntas.",
    "qual é o seu propósito": "Meu propósito é ajudar você a encontrar respostas e a realizar tarefas.",
    "você pode me ajudar a estudar": "Claro! Posso te ajudar com explicações e tirar dúvidas sobre vários assuntos.",
    "me conte uma piada": "Claro! Por que o computador foi ao médico? Porque estava com um vírus!",
    "o que é Python": "Python é uma linguagem de programação de alto nível usada para criar softwares e sistemas.",
    "o que é Java": "Java é uma linguagem de programação muito popular, usada em muitos tipos de aplicativos, como os de celulares e servidores.",
    "qual é a capital do Brasil": "A capital do Brasil é Brasília.",
    "quem é o presidente do Brasil": "O presidente do Brasil é Luiz Inácio Lula da Silva.",
    "qual é a maior cidade do mundo": "A maior cidade do mundo em população é Tóquio, no Japão.",
    "quantos continentes existem": "Existem 7 continentes no mundo: África, Antártida, Ásia, Europa, América do Norte, América do Sul e Oceania.",
    "qual é o animal mais rápido do mundo": "O animal mais rápido do mundo é o falcão-peregrino, que pode atingir até 389 km/h durante um mergulho.",
    "qual é o maior oceano": "O maior oceano do mundo é o Oceano Pacífico.",
    "o que é inteligência artificial": "Inteligência artificial (IA) é a área da computação que desenvolve sistemas capazes de aprender e tomar decisões como seres humanos.",
    "quem foi Albert Einstein": "Albert Einstein foi um físico teórico, famoso por desenvolver a teoria da relatividade.",
    "me diga uma curiosidade": "Você sabia que o cérebro humano é mais ativo quando estamos dormindo do que quando estamos acordados?",
    "o que é machine learning": "Machine learning (aprendizado de máquina) é uma área da inteligência artificial que permite que os sistemas aprendam e melhorem a partir de dados, sem programação explícita.",
    "qual é o significado da vida": "O significado da vida é uma pergunta filosófica que muitas pessoas tentam responder de diferentes maneiras. Para alguns, pode ser encontrar felicidade e propósito.",
    "qual é a maior invenção da humanidade": "Muitas pessoas consideram a invenção da internet como uma das maiores invenções, pois conectou o mundo de uma maneira nunca vista antes.",
    "o que é a teoria da evolução": "A teoria da evolução, proposta por Charles Darwin, sugere que as espécies evoluem ao longo do tempo por meio da seleção natural.",
    "quem inventou a lâmpada": "A lâmpada elétrica foi inventada por Thomas Edison, embora outros inventores também contribuíram para seu desenvolvimento.",
    "qual é o maior animal terrestre": "O maior animal terrestre é o elefante africano.",
    "quem foi Nikola Tesla": "Nikola Tesla foi um inventor, engenheiro elétrico e físico, famoso por suas contribuições ao desenvolvimento do sistema de corrente alternada.",
    "qual é a fórmula da água": "A fórmula química da água é H2O, composta por dois átomos de hidrogênio e um de oxigênio.",
    "qual é o planeta mais próximo do sol": "O planeta mais próximo do Sol é Mercúrio.",
    "quantos países existem no mundo": "Atualmente, existem 195 países reconhecidos no mundo.",
    "quem foi Leonardo da Vinci": "Leonardo da Vinci foi um artista, inventor e cientista renascentista, conhecido por obras como 'A Mona Lisa' e 'A Última Ceia'.",
    "qual é a moeda do Japão": "A moeda do Japão é o iene (JPY).",
    "o que é o aquecimento global": "O aquecimento global é o aumento da temperatura média da Terra devido à emissão de gases de efeito estufa, como dióxido de carbono.",
    "qual é o maior deserto do mundo": "O maior deserto do mundo é o Deserto da Antártida, que é um deserto polar.",
    "qual é o menor país do mundo": "O menor país do mundo em termos de área é o Vaticano, que é um enclave na cidade de Roma.",
    "quantas estrelas tem a bandeira dos Estados Unidos": "A bandeira dos Estados Unidos tem 50 estrelas, representando os 50 estados do país.",
    "o que é o Big Bang": "O Big Bang é a teoria que explica a origem do universo, afirmando que ele começou a partir de uma grande explosão há cerca de 13,8 bilhões de anos.",
    "quem foi Marie Curie": "Marie Curie foi uma cientista que descobriu os elementos rádio e polônio, sendo pioneira no estudo da radioatividade.",
    "qual é o oceano mais profundo do mundo": "O oceano mais profundo do mundo é o Oceano Pacífico, com a Fossa das Marianas sendo o ponto mais profundo.",
    "o que é um buraco negro": "Um buraco negro é uma região do espaço onde a gravidade é tão forte que nada, nem mesmo a luz, pode escapar de sua atração.",
    "o que é uma estrela cadente": "Uma estrela cadente é, na verdade, um meteoro queimando na atmosfera da Terra. O termo 'estrela cadente' é apenas um nome popular.",
    "você sabe programar": "Sim, eu fui programado para entender perguntas e responder com base no que sei.",
    "como posso aprender a programar": "Você pode começar aprendendo linguagens como Python, JavaScript, ou HTML e CSS. Há muitos cursos online gratuitos!",
    "qual é a maior floresta do mundo": "A maior floresta do mundo é a Floresta Amazônica.",
    "quem inventou a luz elétrica": "A invenção da lâmpada elétrica é atribuída a Thomas Edison, embora outros inventores também tenham contribuído.",
    "qual é o animal mais rápido do mundo": "O animal mais rápido do mundo é o falcão-peregrino, que pode atingir velocidades superiores a 380 km/h.",
    "qual é o maior deserto do mundo": "O maior deserto do mundo é o deserto da Antártica, seguido pelo deserto do Saara.",
    "o que é uma nuvem": "Uma nuvem é uma massa de vapor d'água ou cristais de gelo suspensos na atmosfera.",
    "quem foi Isaac Newton": "Isaac Newton foi um matemático e físico inglês, conhecido por suas leis do movimento e a lei da gravitação universal.",
    "o que é um sistema operacional": "Um sistema operacional é o software responsável por gerenciar os recursos do computador e permitir que os programas funcionem.",
    "quem fundou o Google": "O Google foi fundado por Larry Page e Sergey Brin em 1998.",
    "como funciona a energia solar": "A energia solar é gerada por painéis solares que capturam a luz do Sol e a convertem em eletricidade.",
    "qual é a maior montanha do mundo": "A maior montanha do mundo é o Monte Everest, localizado na fronteira entre o Nepal e o Tibete.",
    "o que é o Bitcoin": "O Bitcoin é uma criptomoeda, ou seja, uma moeda digital descentralizada que usa criptografia para segurança.",
    "qual é o objetivo da educação": "O objetivo da educação é ensinar e capacitar os indivíduos para alcançar seu potencial, desenvolver habilidades e contribuir para a sociedade.",
    "como funciona a internet": "A internet é uma rede global de computadores conectados entre si, permitindo a troca de informações e comunicação em tempo real."
}

# Função para preprocessar a pergunta
def preprocessar_pergunta(pergunta):
    pergunta = pergunta.lower()
    pergunta = unidecode(pergunta)  # Remover acentos
    pergunta = pergunta.translate(str.maketrans('', '', string.punctuation))  # Remover pontuação
    tokens = word_tokenize(pergunta)
    stop_words = set(stopwords.words("portuguese"))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# Função para comparar a pergunta com as respostas
def responder(pergunta):
    tokens_pergunta = preprocessar_pergunta(pergunta)
    
    # Verificar a melhor correspondência
    for chave, resposta in perguntas_respostas.items():
        tokens_chave = preprocessar_pergunta(chave)
        similaridade = len(set(tokens_pergunta) & set(tokens_chave)) / len(set(tokens_pergunta) | set(tokens_chave))
        if similaridade > 0.5:
            return resposta
    
    return "Desculpe, não entendi sua pergunta."


# Função para exibir o grafo de conexões
def exibir_grafo():
    G = nx.Graph()
    for chave in perguntas_respostas:
        G.add_node(chave)
    
    for pergunta in perguntas_respostas:
        for outra in perguntas_respostas:
            if pergunta != outra:
                G.add_edge(pergunta, outra)
    
    plt.figure(figsize=(10, 10))
    nx.draw(G, with_labels=True, node_size=7000, node_color="skyblue", font_size=10)
    plt.show()

# Iniciar o chatbot
def iniciar_chatbot():
    print("Bem-vindo ao Chatbot! Digite 'sair' para encerrar a conversa.")
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == 'sair':
            print("Chatbot: Até logo!")
            break
        resposta = responder(pergunta)
        print(f"Chatbot: {resposta}")

# Chamada para iniciar o chatbot
iniciar_chatbot()
