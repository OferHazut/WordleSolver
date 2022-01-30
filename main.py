import numpy as np

letters_p = []


def create_base_range(dec_number, base=3, padding=5):
    padding = max(padding, np.ceil(np.log(dec_number) / np.log(base)))
    out = np.zeros((dec_number, padding), dtype='uint16')
    for i in range(padding):
        out[:, i] = (np.array(range(dec_number)) / (base ** i)).astype('uint16') % base
    return out


def load_dict():
    f = open("Oxford English Dictionary.txt", "r")
    n = f.read()
    s = n.split('\n')
    words = [word.split('  ')[0] for word in s]

    # filter all 5 letters, alphabetical, non-repetitive words
    j = 0
    for i in range(len(words)):
        if len(words[j]) != 5 or not words[j].isalpha() or max([ord(char) > ord('z') for char in words[j].lower()]):
            del words[j]
        else:
            words[j] = words[j].lower()
            j += 1
    return list(np.unique(words))


def get_letters_2d_hist(words):
    counts = np.zeros((26, 26))
    unique_chars_list = []
    for word in words:
        unique_chars_list.append(np.unique([char for char in word.lower()]))
    for i in range(26):
        for j in range(26):
            for chars in unique_chars_list:
                if chr(97 + i) in chars and chr(97 + j) in chars:
                    counts[i, j] += 1
    return counts


def get_letters_1d_hist(words):
    counts = np.zeros(26)
    unique_chars_list = []
    for word in words:
        unique_chars_list.append(np.unique([char for char in word.lower()]))
    for i in range(26):
        for chars in unique_chars_list:
            if chr(97 + i) in chars:
                counts[i] += 1
    return counts


def filter_words(words, letters_list, with_them=True):
    words2 = words.copy()
    for ch in letters_list:
        j = 0
        for i in range(len(words2)):
            if np.logical_xor(ch in [char for char in words2[j].lower()], with_them):
                del words2[j]
            else:
                j += 1
    return words2


def filter_by_place(words, char, place, here=True):
    words2 = words.copy()
    j = 0
    for i in range(len(words2)):
        if np.logical_xor(words2[j][place].lower() == char, here):
            del words2[j]
        else:
            j += 1
    return words2


def compare_word(try_word, real_word):
    real_chars = [char for char in real_word.lower()]
    left_chars = [char for char in real_word.lower()]
    try_chars = [char for char in try_word.lower()]
    answer = np.zeros_like(real_chars, dtype=int)
    for i_char, char in enumerate(real_chars):
        i = answer.size - 1 - i_char
        if try_chars[i] == real_chars[i]:
            answer[i] = 2
            del left_chars[i]
            del try_chars[i]
        elif try_chars[i] in real_chars:
            answer[i] = 1
        else:
            answer[i] = 0

    for i_char, char in enumerate(try_chars):
        if answer[i_char] == 1:
            if char not in left_chars:
                answer[i_char] = -1

    return answer


def filter_by_answer(words, try_word, answer):
    words_copy = words.copy()
    try_chars = [char for char in try_word.lower()]
    for i_char, char in enumerate(try_chars):
        if answer[i_char] == 2:
            words_copy = filter_by_place(words_copy, char, i_char, here=True)
        elif answer[i_char] == 1:
            words_copy = filter_words(words_copy, [char], with_them=True)
            words_copy = filter_by_place(words_copy, char, i_char, here=False)
        elif answer[i_char] == 0:
            words_copy = filter_words(words_copy, [char], with_them=False)
        elif answer[i_char] == -1:
            words_copy = filter_by_place(words_copy, char, i_char, here=False)
    return words_copy


def look_for_best_from_words(words, answer=None):
    if answer is None:
        answer = np.array([0, 0, 0, 0, 0])

    # words list to chars array conv
    chars_list = []
    for word in words:
        chars_list.append([char for char in word.lower()])
    chars_list = np.array(chars_list)

    # delete guessed letters
    for i_num, num in enumerate(answer):
        i = len(answer) - 1 - i_num
        if num == 2:
            chars_list = np.hstack((chars_list[:, :i], chars_list[:, i + 1:]))

    n_words = chars_list.shape[0]
    n_chars = chars_list.shape[1]

    two_arrays = []
    one_vectors = []
    for i in range(26):
        two_array = chars_list == chr(97 + i)
        one_vector = np.sum(two_array, axis=1) > 0
        one_vectors.append(one_vector)  # = np.tile(one_vector, (1, n_chars))
        two_arrays.append(two_array)

    entropies = np.zeros(n_words)
    means = np.zeros(n_words)
    p_s = np.zeros(3 ** n_chars)
    permutes = create_base_range(3 ** n_chars, base=3, padding=n_chars)
    order = np.argsort(np.sum(permutes, axis=1))
    for i_word in range(n_words):
        for i in range(p_s.size):
            union_vec = np.ones_like(entropies, dtype=np.uint8)
            for i_char, char in enumerate(list(chars_list[i_word, :])):
                if permutes[order[i], i_char] == 2:
                    union_vec = np.logical_and(union_vec, two_arrays[ord(char) - 97][:, i_char])
                else:
                    union_vec = np.logical_and(union_vec, np.logical_xor(one_vectors[ord(char) - 97],
                                                                         1 - permutes[order[i], i_char]))

            p_s[i] = np.sum(union_vec) / n_words

        entropies[i_word] = -np.sum(p_s * np.log(p_s + 0.00000001))
        means[i_word] = np.sum(np.array(range(3 ** n_chars)) * p_s)

    # tie-breaker 1: prefer one with more 1s and 2s as an option
    idxs = np.argsort(entropies)[::-1]
    entropies_sorted = np.sort(entropies)[::-1]
    i = 1
    while i < n_words:
        if entropies_sorted[i] == entropies_sorted[0]:
            i += 1
        else:
            break
    if i > 1:
        means = means[idxs][:i]
        chars_list = chars_list[:i]
        j = 1
        while j < i:
            if means[j] == means[0]:
                j += 1
            else:
                break
        # tie-breaker 2: prefer one with more common letters
        if j > 1:
            p = np.zeros(j)
            for k in range(j):
                places = [ord(a) - 97 for a in list(chars_list[k])]
                p[k] = np.prod(letters_p[places])
            return words[idxs[np.argmax(p)]]
        else:
            return words[idxs[np.argmax(means)]]
    else:
        return words[idxs[0]]


if __name__ == '__main__':

    words = load_dict()
    letters_p = get_letters_1d_hist(words)
    trials = []

    for word in words:
        words2 = words.copy()
        try_word = 'stare'
        real_word = word
        trial = 1
        while try_word != real_word:
            # print(try_word)
            answer = compare_word(try_word, real_word)
            # print(str(answer))
            if 1 - min(answer == [2, 2, 2, 2, 2]):
                words2 = filter_by_answer(words2, try_word, answer)
                try_word = look_for_best_from_words(words2, answer)
                trial += 1
        trials.append(trial)
        if len(trials) % 10 == 0:
            print(np.mean(trials))
