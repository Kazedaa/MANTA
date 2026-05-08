from rfdetr import RFDETRBase, RFDETRMedium

model = RFDETRMedium(pretrained=True)

model.train(
    dataset_dir='../uzeta_sot_benchmark/',
    train_ann_file='../uzeta_sot_benchmark/train/_annotations.coco.json',
    val_ann_file='../uzeta_sot_benchmark/valid/_annotations.coco.json',
    output_dir='rfdetr_experiments_12',
    epochs=150,                
    batch_size=32,            
    lr=1e-4,                  
    lr_encoder=1e-5,          
    resolution=640,           
    device='cuda',            
    weight_decay=1e-4,
    amp=True,                 
    grad_accum_steps=4,       
    use_ema=True,             
    checkpoint_interval=1,    
    tensorboard=True,         
    early_stopping=True,
    early_stopping_patience=15,
    early_stopping_min_delta=0.001,
    warmup_epochs=5,
    resume='rfdetr_experiments_12/checkpoint0009.pth'
)

