import os
import re
import random
import chess
import torch
import numpy as np
from scipy import sparse
from tqdm import tqdm


def process_pgn_file(file_path):
    move_pattern = re.compile(r'\b([NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](=[NBRQ])?|O-O(?:-O)?)')
    data = []

    with open(file_path, 'r') as pgn_file:
        content = pgn_file.read()
        moves = move_pattern.findall(content)
        result_match = re.search(r'\s(1-0|0-1|1/2-1/2)\s', content)
        result = result_match.group(1) if result_match else None

        if not moves or not result:
            return []

        board = chess.Board()
        valid_moves = []
        moves = [i for i, _ in moves]

        for move in moves:
            try:
                board.push_san(move)
                valid_moves.append(move)
            except ValueError:
                break

        if valid_moves:
            num_samples = 1 + len(valid_moves) // 15
            sample_indices = random.sample(range(len(valid_moves)), num_samples)

            for sample_index in sample_indices:
                board = chess.Board()
                for i, move in enumerate(valid_moves):
                    if i == sample_index:
                        break
                    board.push_san(move)

                board_state = encode_board(board)
                to_move = 1 if board.turn == chess.WHITE else 0
                next_move = valid_moves[sample_index] if sample_index < len(valid_moves) else None
                next_move2 = valid_moves[sample_index + 2] if sample_index + 2 < len(valid_moves) else None
                is_capture = (("x" in next_move) if next_move else 0) or (("x" in next_move2) if next_move2 else 0)

                if result == '1-0':
                    y = 1
                elif result == '0-1':
                    y = 0
                else:
                    y = 0.5

                if not is_capture:
                    data.append((board_state, to_move, y))
                    if random.random() < 0.0001: print(len(data))

    return data


def encode_board(board):
    rows = []
    cols = []
    data = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = int(piece.color)
            piece_type = piece.piece_type - 1
            rank = square // 8
            file = square % 8
            index = color * 6 * 8 * 8 + piece_type * 8 * 8 + rank * 8 + file
            rows.append(0)  # All in the same row as we're creating a 1D sparse matrix
            cols.append(index)
            data.append(1)
    return sparse.csr_matrix((data, (rows, cols)), shape=(1, 2 * 6 * 8 * 8))


def process_folder(folder_path):
    encodings = []
    to_moves = []
    outcomes = []
    batch = 0

    for root, _, files in tqdm(os.walk(folder_path), desc="Processing folders"):
        for file in files:
            if file.endswith('.pgn'):
                file_path = os.path.join(root, file)
                data = process_pgn_file(file_path)

                for board_state, to_move, outcome in data:
                    encodings.append(board_state)
                    to_moves.append(to_move)
                    outcomes.append(outcome)

                    if len(encodings) == 100000:
                        save_batch(encodings, to_moves, outcomes, batch)
                        batch += 1
                        encodings = []
                        to_moves = []
                        outcomes = []

    if encodings:
        save_batch(encodings, to_moves, outcomes, batch)


def save_batch(encodings, to_moves, outcomes, batch):
    sparse.save_npz(f"encoding_{batch}.npz", sparse.vstack(encodings))

    np.save(f"to_move_{batch}.npy", np.array(to_moves))
    np.save(f"outcomes_{batch}.npy", np.array(outcomes))

    print(f"Saved batch {batch}")


if __name__ == '__main__':
    folder_path = "/home/luka/PycharmProjects/pythonProject/diplomska/nets"
    process_folder(folder_path)
