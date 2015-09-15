import re
import numpy as np


def check_invariants(text):

    print(len(re.findall('by William Shakespeare', text, re.IGNORECASE)))
    print(len(re.findall('Dramatis Personae', text, re.IGNORECASE)))
    print(len(re.findall('\nAct [1i]\.? Scene [1i]', text, re.IGNORECASE)))
    print(len(re.findall('THE END', text)))


def view_random_section(text, length=3000):

    idx = np.random.randint(len(plays) - length)
    print(plays[idx: idx+length])


if __name__ == '__main__':

    with open('speare.txt') as f:
        t = f.read()
        plays_start = re.search('1603', t).start()
        plays_end = re.search('1609\n\nA LOVER\'S COMPLAINT', t).start()

        plays = t[plays_start: plays_end]

    matches = re.finditer('\nAct [1i]\.? Scene [1i].*?THE END', plays, re.DOTALL | re.IGNORECASE)

    plays = '\n\n'.join(m.group() for i, m in enumerate(matches) if i != 3)

    # Get rid of all lines that don't start with whitespace
    plays = re.sub('\n[^ \n].*', '\n', plays)

    # Change formatting of names
    # Eg. 'Romeo. Ho there' becomes 'ROMEO:\nHo there'
    plays = re.sub('\n {1,2}([A-Z][A-Za-z \']+?)\. *', lambda m: '\n{}:\n'.format(m.group(1).upper()), plays)

    # Unindent spoken text
    plays = re.sub('\n {4}([A-Za-z\']+)', lambda m: '\n{}'.format(m.group(1)), plays)

    # Get rid of lines starting with whitespace
    plays = re.sub('\n .*', '\n', plays)

    # Get rid of stage directions that are bracketed eg. [Aside]
    # Note: this uses the fact that all lines now do not start with a space
    # Note: contains lookahead for newline characters (doesn't consume them at the end of the regex)
    # Note: done twice to get rid of lines with two sets of directions
    plays = re.sub('(\n.+?) *(\[.*?\]) *(.*?) *(?=\n)', lambda m: '\n{}{}{}'.format(m.group(1), ' ' if m.group(1) and m.group(3) else '', m.group(3)), plays)
    plays = re.sub('(\n.+?) *(\[.*?\]) *(.*?) *(?=\n)', lambda m: '\n{}{}{}'.format(m.group(1), ' ' if m.group(1) and m.group(3) else '', m.group(3)), plays)

    # Cull stage directions not delimited by brackets but separated by spaces
    plays = re.sub('(\n.+?)  +.*', lambda m: m.group(1), plays)

    # Get rid of multiple newlines back-to-back
    plays = re.sub('\n *(?=\n)', '', plays)

    # view_random_section(plays)

    with open('speare_preproc.txt', 'w') as f:
        f.write(plays)