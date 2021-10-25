import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow
import tflearn

stemmer = LancasterStemmer()

import numpy
import random
import json
import pickle

quotes = [
    "Life is long if you know how to use it.",
    "Living is the least important activity of the preoccupied man; yet there is nothing which is harder to learn.",
    "It does not matter how much time we are given if there is nowhere for it to settle; it escapes through the cracks and holes of the mind.",
    "So it is inevitable that life will be not just very short but very miserable for those who acquire by great toil what they must keep by greater toil.",
    "Of all people only those are at leisure who make time for philosophy, only those are really alive. For they not only keep a good watch over their own lifetimes, but they annex every age to theirs. All the years that have passed before them are added to their own.",
    "But putting things off is the biggest waste of life: it snatches away each day as it comes, and denies us the present by promising the future. The greatest obstacle to living is expectancy, which hangs upon tomorrow and loses today. You are arranging what lies in Fortune’s control, and abandoning what lies in yours. What are you looking at? To what goal are you straining? The whole future lies in uncertainty: live immediately.",
    "You will find no one willing to share out his money; but to how many does each of us divide up his life!",
    "People are frugal in guarding their personal property; but as soon as it comes to squandering time they are most wasteful of the one thing in which it is right to be stingy.",
    "Everyone hustles his life along, and is troubled by a longing for the future and weariness of the present. But the man who spends all his time on his own needs, who organizes every day as though it were his last, neither longs for nor fears the next day.",
    "It is not that we have a short time to live, but that we waste a lot of it. Life is long enough, and a sufficiently generous amount has been given to us for the highest achievements if it were all well invested.",
    "So it is: we are not given a short life but we make it short, and we are not ill-supplied but wasteful of it.",
    "You act like mortals in all that you fear, and like immortals in all that you desire.",
    "How late it is to begin really to live just when life must end! How stupid to forget our mortality, and put off sensible plans to our fiftieth and sixtieth years, aiming to begin life from a point at which few have arrived!",
    "But learning how to live takes a whole life, and, which may surprise you more, it takes a whole life to learn how to die.",
    "You have been preoccupied while life hastens on. Meanwhile death will arrive, and you have no choice in making yourself available for that.",
    "Remember that you’re an actor in a play, which will be as the author chooses, short if he wants it to be short, and long if he wants it to be long. If he wants you to play the part of a beggar, act even that part with all your skill; and likewise if you’re playing a cripple, an official, or a private citizen. For that is your business, to act the role that is assigned to you as well as you can; but it is another’s part to select that role.",
    "Don’t seek that all that comes about should come about as you wish, but wish that everything that comes about should come about just as it does, and then you’ll have a calm and happy life.",
    "…don’t look to what he is doing, but to what you must do if you are to keep your choice in harmony with nature. For no one will cause you harm if you don’t wish it; you’ll have been harmed only when you suppose that you’ve been harmed.",
    "The condition and character of a layman is this: that he never expects that benefit or harm will come to him from himself, but only from externals. The condition and character of a philosopher is this: that he expects all benefit and harm to come to him from himself.",
    "If you regard only that which is your own as being your own, and that which isn’t your own as not being your own (as is indeed the case), no one will ever be able to coerce you, no one will hinder you, you’ll find fault with no one, you’ll accuse no one, you’ll do nothing whatever against your will, you’ll have no enemy, and no one will ever harm you because no harm can affect you.",
    "But for me every omen is favourable for I want it to be so; for whatever may come about, it is within my power to derive benefit from it.",
    "It isn’t the things themselves that disturb people, but the judgements that they form about them.",
    "For it is better to die of hunger, but free from distress and fear, than to live in plenty with a troubled mind.",
    "If someone handed over your body to somebody whom you encountered, you’d be furious; but that you hand over your mind to anyone who comes along, so that, if he abuses you, it becomes disturbed and confused, do you feel no shame at that?",
    "Disease is an impediment to the body, but not to choice, unless choice wills it to be so. Lameness is an impediment to the leg, but not to choice. And tell yourself the same with regard to everything that happens to you; for you’ll find that it acts as an impediment to something else, but not to yourself.",
    "If you want to make progress, put up with being thought foolish and silly with regard to external things, and don’t even wish to give the impression of knowing anything about them.",
    "And even if you’re not yet a Socrates, you ought to live like someone who does in fact wish to be a Socrates.",
    "Remain silent for the most part, or say only what is essential, and in few words.",
    "Never call yourself a philosopher, and don’t talk among laymen for the most part about philosophical principles, but act in accordance with those principles…And accordingly, if any talk should arise among laymen about some philosophical principle, keep silent for the most part, for there is a great danger that you’ll simply vomit up what you haven’t properly digested.",
    "In each action that you undertake, consider what comes before and what follows after, and only then proceed to the action itself.",
    "In things relating to the body, take only as much as your bare need requires, with regard to food, for instance, or drink, clothes, housing … exclude everything that is for show or luxury.",
    "He does only what is his to do, and considers constantly what the world has in store for him—doing his best, and trusting that all is for the best. For we carry our fate with us —and it carries us.",
    "To love only what happens, what was destined. No greater harmony.",
    "For there is a single harmony. Just as the world forms a single body comprising all bodies, so fate forms a single purpose, comprising all purposes.",
    "To watch the courses of the stars as if you revolved with them. To keep constantly in mind how the elements alter into one another. Thoughts like this wash off the mud of life below.",
    "No one can keep you from living as your nature requires. Nothing can happen to you that is not required by Nature.",
    "The others obey their own lead, follow their own impulses. Don’t be distracted. Keep walking. Follow your own nature, and follow Nature—along the road they share.",
    "Doctors keep their scalpels and other instruments handy, for emergencies. Keep your philosophy ready too—ready to understand heaven and earth. In everything you do, even the smallest thing, remember the chain that links them. Nothing earthly succeeds by ignoring heaven, nothing heavenly by ignoring the earth.",
    "Love the discipline you know, and let it support you. Entrust everything willingly to the gods, and then make your way through life—no one’s master and no one’s slave.",
    "People try to get away from it all—to the country, to the beach, to the mountains. You always wish that you could too. Which is idiotic: you can get away from it anytime you like. By going within. Nowhere you can go is more peaceful—more free of interruptions—than your own soul.",
    "Whatever happens to you is for the good of the world. That would be enough right there. But if you look closely you’ll generally notice something else as well: whatever happens to a single person is for the good of others.",
    "In short, know this: Human lives are brief and trivial. Yesterday a blob of semen; tomorrow embalming fluid, ash.",
    "Indifference to external events. And a commitment to justice in your own acts. Which means: thought and action resulting in the common good. What you were born to do.",
    "When you wake up in the morning, tell yourself: The people I deal with today will be meddling, ungrateful, arrogant, dishonest, jealous, and surly. They are like this because they can’t tell good from evil. But I have seen the beauty of good, and the ugliness of evil, and have recognized that the wrongdoer has a nature related to my own—not of the same blood or birth, but the same mind, and possessing a share of the divine.",
    "Fight to be the person philosophy tried to make you. Revere the gods; watch over human beings. Our lives are short. The only rewards of our existence here are an unstained character and unselfish acts.",
    "Mastery of reading and writing requires a master. Still more so life.",
    "People who labor all their lives but have no purpose to direct every thought and impulse toward are wasting their time—even when hard at work.",
    "At dawn, when you have trouble getting out of bed, tell yourself: ‘I have to go to work—as a human being. What do I have to complain of, if I’m going to do what I was born for— the things I was brought into the world to do? Or is this what I was created for? To huddle under the blankets and stay warm?",
    "You don’t love yourself enough. Or you’d love your nature too, and what it demands of you. People who love what they do wear themselves down doing it, they even forget to wash or eat.",
    "It can ruin your life only if it ruins your character. Otherwise it cannot harm you—inside or out.",
    "Perfection of character: to live your last day, every day, without frenzy, or sloth, or pretense.",
    "The things you think about determine the quality of your mind. Your soul takes on the color of your thoughts.",
    "The mind in itself has no needs, except for those it creates itself. Is undisturbed, except for its own disturbances. Knows no obstructions, except those from within.",
    "The mind without passions is a fortress. No place is more secure. Once we take refuge there we are safe forever. Not to see this is ignorance. To see it and not seek safety means misery.",
    "That things have no hold on the soul. They stand there unmoving, outside it. Disturbance comes only from within—from our own perceptions.",
    "Choose not to be harmed—and you won’t feel harmed. Don’t feel harmed and you haven’t been.",
    "Objective judgment, now, at this very moment. Unselfish action, now, at this very moment. Willing acceptance—now, at this very moment—of all external events. That’s all you need.",
    "I can control my thoughts as necessary; then how can I be troubled? What is outside my mind means nothing to it. Absorb that lesson and your feet stand firm. You can return to life. Look at things as you did before. And life returns.",
    "External things are not the problem. It’s your assessment of them. Which you can erase right now.",
    "Your three components: body, breath, mind. Two are yours in trust; to the third alone you have clear title.",
    "To move from one unselfish action to another with God in mind. Only there, delight and stillness.",
    "For every action, ask: How does it affect me? Could I change my mind about it? But soon I’ll be dead, and the slate’s empty. So this is the only question: Is it the action of a responsible being, part of society, and subject to the same decrees as God?",
    "Stop talking about what the good man is like, and just be one.",
    "Be satisfied with even the smallest progress, and treat the outcome of it all as unimportant.",
    "You see how few things you have to do to live a satisfying and reverent life? If you can manage this, that’s all even the gods can ask of you.",
    "‘If you seek tranquillity, do less.’ Or (more accurately) do what’s essential, what the logos of a social being requires, and in the requisite way. Which brings a double satisfaction: to do less, better. Because most of what we say and do is not essential. If you can eliminate it, you’ll have more time, and more tranquillity. Ask yourself at every moment, ‘Is this necessary?'",
    "The impediment to action advances action. What stands in the way becomes the way.",
    "To bear in mind constantly that all of this has happened before. And will happen again—the same plot from beginning to end, the identical staging. Produce them in your mind, as you know them from experience or from history…All just the same. Only the people different.",
    "Forget everything else. Keep hold of this alone and remember it: Each of us lives only now, this brief instant. The rest has been lived already, or is impossible to see.",
    "Think of yourself as dead. You have lived your life. Now take what’s left and live it properly.",
]
others = [
    "Of all people only those are at leisure who make time for philosophy, only those are really alive. For they not only keep a good watch over their own lifetimes, but they annex every age to theirs. All the years that have passed before them are added to their own.",
    "The condition and character of a layman is this: that he never expects that benefit or harm will come to him from himself, but only from externals. The condition and character of a philosopher is this: that he expects all benefit and harm to come to him from himself.",
    "If someone handed over your body to somebody whom you encountered, you’d be furious; but that you hand over your mind to anyone who comes along, so that, if he abuses you, it becomes disturbed and confused, do you feel no shame at that?",
    "And even if you’re not yet a Socrates, you ought to live like someone who does in fact wish to be a Socrates.",
    "Remain silent for the most part, or say only what is essential, and in few words.",
    "Never call yourself a philosopher, and don’t talk among laymen for the most part about philosophical principles, but act in accordance with those principles…And accordingly, if any talk should arise among laymen about some philosophical principle, keep silent for the most part, for there is a great danger that you’ll simply vomit up what you haven’t properly digested.",
    "In each action that you undertake, consider what comes before and what follows after, and only then proceed to the action itself.",
    "In things relating to the body, take only as much as your bare need requires, with regard to food, for instance, or drink, clothes, housing … exclude everything that is for show or luxury.",
    "The others obey their own lead, follow their own impulses. Don’t be distracted. Keep walking. Follow your own nature, and follow Nature—along the road they share.",
    "Doctors keep their scalpels and other instruments handy, for emergencies. Keep your philosophy ready too—ready to understand heaven and earth. In everything you do, even the smallest thing, remember the chain that links them. Nothing earthly succeeds by ignoring heaven, nothing heavenly by ignoring the earth.",
    "Love the discipline you know, and let it support you. Entrust everything willingly to the gods, and then make your way through life—no one’s master and no one’s slave.",
    "Whatever happens to you is for the good of the world. That would be enough right there. But if you look closely you’ll generally notice something else as well: whatever happens to a single person is for the good of others.",
    "When you wake up in the morning, tell yourself: The people I deal with today will be meddling, ungrateful, arrogant, dishonest, jealous, and surly. They are like this because they can’t tell good from evil. But I have seen the beauty of good, and the ugliness of evil, and have recognized that the wrongdoer has a nature related to my own—not of the same blood or birth, but the same mind, and possessing a share of the divine.",
    "Fight to be the person philosophy tried to make you. Revere the gods; watch over human beings. Our lives are short. The only rewards of our existence here are an unstained character and unselfish acts.",
    "Stop talking about what the good man is like, and just be one.",
    "Your three components: body, breath, mind. Two are yours in trust; to the third alone you have clear title.",
    "For every action, ask: How does it affect me? Could I change my mind about it? But soon I’ll be dead, and the slate’s empty. So this is the only question: Is it the action of a responsible being, part of society, and subject to the same decrees as God?",
]


with open("intents.json") as file:
    data = json.load(file)

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print(
        """

    ###############################################################
          _____   _             _          ____            _   
         / ____| | |           (_)        |  _ \          | |  
        | (___   | |_    ___    _    ___  | |_) |   ___   | |_ 
         \___ \  | __|  / _ \  | |  / __| |  _ <   / _ \  | __|
         ____) | | |_  | (_) | | | | (__  | |_) | | (_) | | |_ 
        |_____/   \__|  \___/  |_|  \___| |____/   \___/   \__|    

    ###############################################################


    Welcome, I am StoicBot!
    I will respond to any concerns you have with a quote from one of the great stoics
    Please describe your emotions to help me advise you
    Type "quit" to exit
    """
    )
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.4:
            for block in data["intents"]:
                if block["tag"] == tag:
                    responses = block["responses"]
                    break

            print("StoicBot:", random.choice(responses), "\n")
        else:
            print("StoicBot: ", random.choice(quotes), "\n")


chat()
