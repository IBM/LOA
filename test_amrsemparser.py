import os

from amr_parser import get_verbnet_preds_from_obslist

amr_server_ip = os.getenv('LOA_AMR_SERVER_IP', 'localhost')
amr_server_port = int(os.getenv('LOA_AMR_SERVER_PORT', '0'))

all_preds, pred_count_dict, verbnet_facts_logs = \
    get_verbnet_preds_from_obslist(
        # obslist=['You pick up the wet hoodie from the ground. ' +
        #          'You pick up an red apple from the black big table.'],
        # obslist=['You pick up the wet hoodie from the ground and ' +
        #          'you pick up an apple from the table'],
        obslist=[
            '-= Backyard =- ' +
            'I just think it\'s great that you\'ve just entered a backyard. ' +
            'I guess you better just go and list everything you see here. ' +
            'You can make out a BBQ. The BBQ is recent. ' +
            'But the thing hasn\'t got anything on it. ' +
            'What you think everything in TextWorld should have stuff on it? ' +
            'You make out a clothesline. The clothesline is typical. ' +
            'But the thing is empty, unfortunately. ' +
            'Aw, here you were, all excited for there to be things on it! ' +
            'As if things weren\'t amazing enough already, ' +
            'you can even see a patio chair. The patio chair is stylish. ' +
            'However, the patio chair, like an empty patio chair, ' +
            'has nothing on it. You make out a patio table. ' +
            'The patio table is stylish. But there isn\'t a thing on it. ' +
            'You bend down to tie your shoe. When you stand up, ' +
            'you notice a workbench. ' +
            'On the workbench you see a wet cardigan. ' +
            'There is an open screen door leading west.'
        ],
        amr_server_ip=amr_server_ip,
        amr_server_port=amr_server_port,
        mincount=0, verbose=True,
        sem_parser_mode='both',
    )

# print(all_preds)
# print(pred_count_dict)
print(verbnet_facts_logs)
