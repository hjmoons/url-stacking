# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou
# https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py

import random
import numpy as np

random.seed(33)

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################


def random_swap(domains, n):
    new_domains = domains.copy()
    for _ in range(n):
        new_domains = swap_domain(new_domains)

    return new_domains


def swap_domain(new_domains):
    # swap in second(third)-level domain
    # for example 'google.co.kr', we use only 'google' to swap

    # to search first start location(not padding)
    start_domain = 0

    # to search dot(.) location
    dot_domain = 0

    for c in new_domains:
        if c == 0:
            start_domain = start_domain + 1

        # when found '.'('.' is 4 in tokenizer), break
        if c == 4:
            break

        dot_domain = dot_domain + 1

    random_idx_1 = random.randint(start_domain, dot_domain-1)
    random_idx_2 = random_idx_1
    counter = 0

    # swap in before '.'
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(start_domain, dot_domain-1)
        counter += 1
        if counter > 3:
            return new_domains

    new_domains[random_idx_1], new_domains[random_idx_2] = new_domains[random_idx_2], new_domains[random_idx_1]

    return new_domains


def eda(domain, alpha_rs=0.1, num_aug=3):
    num_chars = 40

    augmented_domains = np.array(domain).reshape(1, 74)
    # print(augmented_domains)
    num_new_per_technique = int(num_aug / 1) + 1    # int(num_aug / 4) + 1
    n_rs = max(1, int(alpha_rs * num_chars))

    # rs
    for _ in range(num_new_per_technique):

        a_domains = random_swap(domain, n_rs)
        a_domains = np.array(a_domains).reshape(1, 74)
        augmented_domains = np.append(augmented_domains, a_domains, axis=0)

    augmented_domains = augmented_domains[1:]

    if num_aug >= 1:
        augmented_domains = augmented_domains[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_domains)
        augmented_domains = [s for s in augmented_domains if random.uniform(0, 1) < keep_prob]

    augmented_domains = np.append(augmented_domains, np.array(domain).reshape(1, 74), axis=0)

    return augmented_domains
